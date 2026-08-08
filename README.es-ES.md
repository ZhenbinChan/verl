

# verl para RL de Búsqueda en Árbol

Este repositorio es un fork de `verl` centrado en el entrenamiento de RL para tareas de razonamiento, con soporte adicional para la expansión de rollout en estilo de árbol, recompensa de proceso a nivel de paso, verificación FOL/Z3 y conjuntos de datos de QA lógico como LogiQA y ReClor.

## Qué se ha añadido

- Estrategias de expansión de rollout enchufables bajo `trainer.sampling_strategy`: rollout plano, búsqueda en árbol legada, TreeRL de cadena de entropía, MCTS paralelo, Step-TreeRL y expansión por ganancia de información.
- Recompensa de proceso a nivel de paso a través de `trainer.process_reward.type=format|fol`, compartida por Step-TreeRL, MCTS paralelo y muestreo por ganancia de información.
- Utilidades FOL-as-PRM para traducir/verificar pasos de razonamiento lógico con un backend de LLM compatible con OpenAI, MiniMax o Azure OpenAI, además de Z3.
- Preprocesadores de conjuntos de datos y scripts de lanzamiento para GSM8K, LogiQA, ReClor, datos estilo MCQ, metadata FOL, GRPO y experimentos Step-TreeRL.
- Métricas de entrenamiento adicionales para rollouts en árbol, especialmente conteos de trazas de Step-TreeRL, precisión de hojas, ratio de formato y timing.

## Estructura del repositorio

- `verl/trainer/config/ppo_trainer.yaml`: punto de entrada principal de configuración PPO/GRPO.
- `verl/trainer/ppo/sampling/`: implementaciones de estrategias de expansión de rollout.
- `verl/workers/reward_manager/`: gestores de recompensa para flujos de trabajo planos, de árbol, entropía, MCTS, Step-TreeRL y ganancia de información.
- `verl/utils/process_reward.py`: configuración y constructor de tiempo de ejecución canónico para recompensa de proceso.
- `examples/data_preprocess/`: scripts de preprocesamiento de conjuntos de datos y metadata FOL.
- `bash_scripts/logiqa/`, `bash_scripts/reclor/`, `bash_scripts/TreeSearch/`: scripts de experimento ejecutables.
- `CONFIG.md`: guía de configuración detallada.

## Instalación

### Requisitos previos

- Python >= 3.11
- CUDA >= 12.4
- PyTorch 2.6.0 recomendado
- Ray 2.48.0
- vLLM 0.8.5.post1 recomendado

### Configuración

```bash
conda create -n verl_plus python=3.11
conda activate verl_plus

git clone https://github.com/BiNLP/verl
cd verl
pip install -e .
pip install -r requirements.txt
```

Pila recomendada CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5.post1
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## Preparación de conjuntos de datos

### GSM8K

```bash
python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
```

### LogiQA / ReClor

```bash
python3 examples/data_preprocess/logiqa.py --local_dir data/logiqa
python3 examples/data_preprocess/reclor.py --local_dir data/reclor
```

### Preprocesador MCQ

Usa `mcq_preprocess.py` al convertir archivos parquet de opción múltiple existentes o al preparar metadata FOL.

```bash
python examples/data_preprocess/mcq_preprocess.py \
  --input_parquet data/reclor/train.parquet \
  --output_dir data/reclor \
  --preset reclor \
  --skip_fol_extraction
```

Para metadata FOL-as-PRM:

```bash
python examples/data_preprocess/mcq_preprocess.py \
  --input_parquet data/reclor/train.parquet \
  --output_dir data/reclor_fol \
  --preset reclor \
  --base_url "http://localhost:4869/v1" \
  --model "qwen2.5-7b-coder" \
  --max_retries -1 \
  --verbose
```

Salidas esperadas:

- `train.parquet` y `test.parquet`: datos de entrenamiento/validación.
- `fol_metadata.json`: requerido cuando `trainer.process_reward.type=fol` y se usa metadata offline.

### Metadata Global de FOL PRM

Usa el preprocesador de metadata por splits al preparar datos FOL PRM para Step-TreeRL. Convierte los splits del conjunto de datos por separado y escribe metadata específica para cada split y un archivo de metadata fusionado.

```bash
bash bash_scripts/preprocess/global_fol_prm_metadata_splits.sh \
  --api_config llm_server/configs/deepseek.yaml
```

El script usa ReClor por defecto:

- input: `data/reclor`
- output: `data/reclor_global_fol_prm`
- splits: descubiertos de `train.parquet`, `test.parquet` y alias de validación si están presentes
- metadata: `fol_metadata_train.json`, `fol_metadata_test.json` y fusionado `fol_metadata_all.json`

Para una prueba inicial de API:

```bash
bash bash_scripts/preprocess/global_fol_prm_metadata_splits.sh \
  --api_config llm_server/configs/deepseek.yaml \
  --output_dir /tmp/verl_fol_deepseek_reclor_smoke \
  --splits train,test \
  --num_samples_per_split 1 \
  --max_workers 1 \
  --max_retries 3 \
  --save_every 1
```

La configuración del proveedor se carga desde archivos YAML o JSON dentro de `llm_server/configs/`. Por ejemplo, `llm_server/configs/deepseek.yaml` contiene el endpoint compatible con OpenAI, el nombre del modelo, los valores predeterminados de solicitud, `extra_body` opcional y el campo `api_key` utilizado tanto en el preprocesamiento como en la verificación FOL en línea.

## Modos de entrenamiento

El punto de entrada por defecto es:

```bash
python3 -m verl.trainer.main_ppo key=value ...
```

### GRPO Plano

Úsalo para entrenamiento de recompensa a nivel de respuesta normal sin expansión en árbol.

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.rollout.n=4 \
  reward_model.reward_manager=auto \
  trainer.sampling_strategy=null
```

Scripts de ejemplo:

- `bash_scripts/logiqa/Qwen3-8B-base_GRPO_base.sh`
- `bash_scripts/reclor/Qwen3-8B-base_GRPO_base.sh`
- `examples/grpo_trainer/run_qwen2-7b.sh`

### Step-TreeRL

Úsalo para expansión en árbol a nivel de paso: genera rollouts iniciales completos, divídelos por `<step>...</step>`, selecciona nodos de paso de alta entropía, ramifica desde los nodos seleccionados, retropropaga la corrección/valor de las hojas y luego entrena sobre las trazas terminales seleccionadas.

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=step_treerl_grpo \
  actor_rollout_ref.actor.policy_loss=tree_loss \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.rollout.n=6 \
  reward_model.reward_manager=auto \
  trainer.sampling_strategy=step_treerl \
  trainer.process_reward.type=format \
  trainer.step_treerl_config.m=6 \
  trainer.step_treerl_config.n=2 \
  trainer.step_treerl_config.l=1 \
  trainer.step_treerl_config.t=2 \
  trainer.step_treerl_config.selected_num_traces=16
```

Scripts de ejemplo:

- `bash_scripts/logiqa/Qwen3-8B-base_StepTreeRL_format_reward.sh`
- `bash_scripts/logiqa/Qwen3-8B-base_StepTreeRL_fol_reward.sh`
- `bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_format_reward.sh`
- `bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_fol_reward.sh`

### Otras estrategias de árbol

| Estrategia | `trainer.sampling_strategy` | Gestor de recompensa | Estimador de ventaja | Configuración principal |
| --- | --- | --- | --- | --- |
| Búsqueda en árbol legada | `tree_search` | `tree` | `tree_grpo` o `tree_gae` | `trainer.tree_rounds`, `tree_top_k`, `branch_level` |
| TreeRL de cadena de entropía | `treerl` | `entropy` | `entropy_reinforce` | `trainer.entropy_chain_config` |
| MCTS paralelo | `parallel_mcts` | `mcts` | `mcts_grpo` | `trainer.parallel_mcts_config` |
| Step-TreeRL | `step_treerl` | `step_tree` | `step_treerl_grpo` o `step_treerl_reinforce` | `trainer.step_treerl_config` |
| Ganancia de información | `information_gain` | `ig` | `ig_grpo` | `trainer.ig_config` |

`reward_model.reward_manager=auto` resuelve el gestor a partir de `trainer.sampling_strategy` para las estrategias de árbol anteriores.

## Recompensa de Proceso

La recompensa de proceso reside bajo `trainer.process_reward`, no en `reward_model.reward_kwargs`.

### PRM de Formato

```bash
trainer.process_reward.type=format
```

Esto verifica si los pasos de razonamiento siguen el formato de paso esperado. Es la recompensa de proceso más simple y suele ser suficiente para pruebas iniciales.

## Métricas de Formato de Rollout

Las métricas de formato de rollout son diagnósticos a nivel de entrenador para prompts que solicitan al modelo emitir
Los scripts GRPO de LogiQA también exponen esto como:

```bash
LOG_FORMAT_METRICS=True bash bash_scripts/logiqa/Qwen2.5-7B_LogiQA_GRPO_only.sh
```

Estas métricas se calculan después de que la expansión de rollout haya finalizado y después de que las trayectorias de entrenamiento finales hayan sido ensambladas. Esto significa que se aplican a rollouts de GRPO plano y a trazas expandidas de `tree_search`, `treerl`, `parallel_mcts`, `step_treerl` e `information_gain`. Son independientes del gestor de recomp`Para la validación con `naive_plus` o `step_tree`, los detalles de formato se registran como campos numéricos derivados de `classify_rollout_format` como `format_primary_full`, `format_primary_boxed_missing`, `boxed_status_valid`, `relaxed_format_correct`, `step_block_count` y `format_error_advantage_mask`. Estos aparecen como métricas `val-aux/{data_source}/...`.`

### Métricas de Entrenamiento

| Campo | Significado |
| --- | --- |
| `rollout/format_primary/total` | Número de trayectorias de rollout incluidas en las estadísticas de formato para este paso de entrenamiento. |
| `rollout/format_primary/full_ratio` | Fracción de trayectorias cuyos bloques de paso y respuesta final en caja son válidas. |
| `rollout/format_primary/relax_correct_ratio` | Fracción cuya XML/esquema de paso y respuesta final en caja son válidos al ignorar texto arbitrario fuera de los bloques de paso completos. |
| `rollout/format_primary/no_step_ratio` | Fracción de trayectorias sin bloque `<step>...</step>` en la región de razonamiento. |
| `rollout/format_primary/text_outside_step_ratio` | Fracción de trayectorias que contienen bloques de paso completos, pero también contienen texto no blanco fuera de esos bloques de paso antes de la respuesta final. |
| `rollout/format_primary/step_xml_invalid_ratio` | Fracción de trayectorias con XML de paso malformado, como una etiqueta `<step>` abierta sin un bloque de cierre válido. |
| `rollout/format_primary/step_schema_invalid_ratio` | Fracción de trayectorias cuyo XML de paso se analiza, pero el esquema de paso es inválido. |
| `rollout/format_primary/boxed_missing_ratio` | Fracción de trayectorias con bloques de paso válidos pero sin respuesta final `\boxed{...}`. |
| `rollout/format_primary/boxed_invalid_ratio` | Fracción de trayectorias con bloques de paso válidos y una región de respuesta `\boxed`, pero el formato de respuesta es inválido. |
| `rollout/answer_acc/all_correct_ratio` | Precisión de respuesta sobre todas las trayectorias de rollout cuya corrección de respuesta puede leerse del gestor de recomp` |
| `rollout/answer_acc/format_correct_only_ratio` | Precisión de respuesta después de eliminar trayectorias incorrectas de formato; esto se calcula solo sobre trayectorias con `format_primary=full`. |

Las proporciones de categoría principal son mutuamente excluyentes y suman 1 para un lote de rollout no vacío. `relax_correct_ratio` es una métrica derivada y no forma parte de esa suma.

`reward/mean_fn_reward` es la recompensa final de entrenamiento, mientras que `rollout/answer_acc/...` registra la corrección de respuesta. Para `naive_plus`, las trayectorias incorrectas de formato reciben `-1` de recompensa en el último token de respuesta válido solo cuando `reward_model.reward_kwargs.penalize_format_error=True`, por lo que `reward/mean_fn_reward` puede diferir de la precisión de respuesta. El valor por defecto es `False` para preservar el comportamiento de entrenamiento de `prompts/base.txt`.

```bash
reward_model.reward_kwargs.penalize_format_error=True
```

### Reglas de Formato

Una trayectoria se cuenta como `full` solo cuando se cumplen todas las siguientes condiciones:

- La región de razonamiento contiene uno o más bloques `<step>...</step>` completos.
- No hay texto fuera de los bloques de paso antes de la respuesta final, excepto espacio real o las secuencias de escape literales `\n`, `\r`, `\t`, `\v` y `\f`.
- Cada paso es XML válido con la etiqueta raíz `step`.
- Cada paso contiene al menos un `<premise>...</premise>` y exactamente un `<conclusion>...</conclusion`.
- Los hijos de paso son solo `premise` o `conclusion`; etiquetas anidadas, texto directo bajo `<step>` y colas de hijo no blancas son inválidas.
- La respuesta final es una respuesta estricta de caja final con exactamente un índice alfabético, por ejemplo `\boxed{A}` o `\boxed{{A}}`.

Ejemplos de respuestas en caja inválidas incluyen `\boxed{A}}`, `\boxed{{A}`, `\boxed{AA}`, `\boxed{AB}`, `\boxed{1}` y una caja vacía.

Para `relax_correct_ratio`, el texto arbitrario fuera de los bloques de paso completos se ignora, pero aún se requiere al menos un paso completo, XML/esquema de paso válido, sin etiquetas de paso sin emparejar y una respuesta de caja final válida. Las secuencias de escape literales de espacio después de la respuesta en caja no se ignoran porque la respuesta en caja debe permanecer como el final estricto de la respuesta.

### Campos de JSONL de Rollout

Cuando `trainer.rollout_data_dir` está configurado y `trainer.log_format_metrics=True`, las filas JSONL de rollout volcadas incluyen estos campos por trayectoria:

| Campo | Significado |
| --- | --- |
| `format_primary` | La categoría mutuamente exclusiva asignada a esta trayectoria: `full`, `no_step`, `text_outside_step`, `step_xml_invalid`, `step_schema_invalid`, `boxed_missing` o `boxed_invalid`. |
| `boxed_status` | `valid`, `invalid` o `missing`, describiendo solo la región de respuesta final en caja. |
| `boxed_answer` | La letra de respuesta extraída cuando `boxed_status=valid`; de lo contrario, una cadena vacía. |
| `step_block_count` | Número de bloques `<step>...</step>` completos encontrados antes de la región de respuesta final. |
| `format_error_advantage_mask` | `0.0` cuando `format_primary=full`, de lo contrario `1.0`. Si `algorithm.mask_format_error_advantage=True`, las filas con `1.0` tienen sus ventajas puestas a cero. |
| `answer_acc` | Corrección de respuesta por trayectoria, cuando sea proporcionada por el gestor de recomp` |

### PRM FOL

```bash
trainer.process_reward.type=fol \
trainer.process_reward.fol.prm_mode=global_fol_prm \
trainer.process_reward.fol.metadata_path=/path/to/fol_metadata.json \
trainer.process_reward.fol.llm.api_config=llm_server/configs/deepseek.yaml
```

La puntuación FOL necesita metadata `sample_id` en el lote. Para FOL PRM global, usa el archivo de metadata de split fusionado como `data/reclor_global_fol_prm/fol_metadata_all.json`, para que los IDs de train, test y validation se resuelvan de manera consistente. Si la metadata falta y `online_declaration_fallback=true`, el runtime puede generar declaraciones en línea a través del backend de LLM configurado.

La configuración FOL LLM aún puede ser sobreescrita desde la línea de comandos, pero la ruta recomendada es mantener la configuración específica del proveedor en `llm_server/configs/*.yaml` y pasar solo `trainer.process_reward.fol.llm.api_config=...` en los scripts de entrenamiento.

### Script FOL Step-TreeRL para ReClor

El script principal de ReClor FOL Step-TreeRL es:

```bash
bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_fol_reward.sh
```

Utiliza `FOL_API_CONFIG` para seleccionar la configuración del proveedor. Ejemplo:

```bash
FOL_API_CONFIG=/home/chenzhb/Workspaces/verl/llm_server/configs/deepseek.yaml \
bash bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_fol_reward.sh
```

La prueba inicial de 1 paso reciente usó el mismo script con sobrescrituras:

```bash
FOL_API_CONFIG=/home/chenzhb/Workspaces/verl/llm_server/configs/deepseek.yaml \
bash bash_scripts/reclor/Qwen3-8B-base_StepTreeRL_fol_reward.sh \
  actor_rollout_ref.model.path=/home/chenzhb/Workspaces/LLMs/Qwen2.5-1.5B-Instruct \
  data.train_files=/tmp/verl_fol_deepseek_reclor_direct_key_train2/train.parquet \
  data.val_files=/tmp/verl_fol_deepseek_reclor_direct_key_train2/test.parquet \
  data.train_batch_size=2 \
  data.max_prompt_length=1024 \
  data.max_response_length=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_model_len=1536 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.step_treerl_config.max_depth=1 \
  trainer.step_treerl_config.max_token_num=256 \
  trainer.step_treerl_config.branch_max_new_tokens=64 \
  trainer.step_treerl_config.m=1 \
  trainer.step_treerl_config.n=1 \
  trainer.step_treerl_config.l=0 \
  trainer.step_treerl_config.t=1 \
  trainer.step_treerl_config.selected_num_traces=1 \
  trainer.process_reward.fol.metadata_path=/tmp/verl_fol_deepseek_reclor_direct_key_train2/fol_metadata_all.json \
  trainer.process_reward.fol.max_retries=1 \
  trainer.process_reward.fol.verify_timeout=10 \
  trainer.process_reward.fol.llm.max_concurrency=1 \
  trainer.logger="['console']" \
  trainer.experiment_name=StepTreeRL_Reclor_FOL_deepseek_direct_key_smoke \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1
```

En esa ejecución, el paso de entrenamiento se completó y registró `reward/step_treerl_process_reward_mean`. El guardado final del checkpoint falló porque el sistema de archivos del espacio de trabajo estaba lleno, no por la metadata FOL o la carga de la configuración del proveedor.

## Evaluación

Usa los scripts de evaluación integrados:

```bash
sh bash_scripts/eval/eval_lighteval.sh
sh bash_scripts/eval/eval_QA_lighteval.sh
```

Ayudantes específicos de conjunto de datos:

- `bash_scripts/eval/Qwen2.5-1.5B_LogiQA_eval.sh`
- `bash_scripts/eval/Qwen2.5-1.5B_ReClor_eval.sh`
- `bash_scripts/eval/Qwen2.5-7B_LogiQA_eval.sh`
- `bash_scripts/eval/Qwen2.5-7B_ReClor_eval.sh`

### Evaluación MCQ Cruzada de Dominio

La evaluación cruzada de dominio utiliza archivos parquet locales más el flujo de evaluación de dos etapas del repositorio:

1. `verl.trainer.main_generation` genera respuestas del modelo en `eval_output/main_eval/.../*_generated.parquet`.
2. `verl.trainer.main_eval` puntúa las respuestas generadas con `bash_scripts/eval/custom_module.py`, que delega en `verl.utils.reward_score.default_compute_score`.

Los scripts actuales de Qwen3-8B-Base usan:

```bash
MODEL_PATH=/home/chenzhb/Workspaces/LLMs/Qwen3-8B-Base
```

Prepara o actualiza los archivos parquet de dominio cruzado:

```bash
python3 examples/data_preprocess/pubmedqa.py \
  --data_dir ./data/pubmedqa_origin/data \
  --local_dir ./data/pubmedqa/

python3 examples/data_preprocess/truthfulqa.py \
  --local_dir ./data/truthfulqa/

python3 examples/data_preprocess/qa4mre.py \
  --local_dir ./data/qa4mre/

python3 examples/data_preprocess/gpqa.py \
  --local_dir ./data/gpqa/

python3 examples/data_preprocess/mathqa.py \
  --data_dir ./data/MathQA \
  --local_dir ./data/mathqa/

python3 examples/data_preprocess/openbookqa.py \
  --local_dir ./data/openbookqa/

python3 examples/data_preprocess/medqa.py \
  --local_dir ./data/medqa/
```

Archivos de evaluación esperados:

| Conjunto de datos | Parquet |
| --- | --- |
| PubMedQA | `data/pubmedqa/test.parquet` |
| TruthfulQA MC1 | `data/truthfulqa/test.parquet` |
| QA4MRE 2013 EN | `data/qa4mre/test.parquet` |
| GPQA Diamond | `data/gpqa/gpqa_diamond/test.parquet` |
| GPQA Main | `data/gpqa/gpqa_main/test.parquet` |
| MathQA | `data/mathqa/test.parquet` |
| MathQA Challenge | `data/mathqa/challenge_test.parquet` |
| OpenBookQA | `data/openbookqa/test.parquet` |
| MedQA | `data/medqa/test.parquet` |

Ejecuta uno de los scripts de evaluación Qwen3-8B-Base específicos de conjunto de datos:

```bash
bash bash_scripts/eval/qwen3-8b-base_pubmedqa_eval.sh
bash bash_scripts/eval/qwen3-8b-base_truthfulqa_eval.sh
bash bash_scripts/eval/qwen3-8b-base_qa4mre_eval.sh
bash bash_scripts/eval/qwen3-8b-base_gpqa_diamond_eval.sh
bash bash_scripts/eval/qwen3-8b-base_gpqa_main_eval.sh
bash bash_scripts/eval/qwen3-8b-base_mathqa_eval.sh
bash bash_scripts/eval/qwen3-8b-base_mathqa_challenge_eval.sh
bash bash_scripts/eval/qwen3-8b-base_openbookqa_eval.sh
bash bash_scripts/eval/qwen3-8b-base_medqa_eval.sh
```

El ejecutable compartido también puede llamarse directamente con un nombre de conjunto de datos:

```bash
bash bash_scripts/eval/qwen3-8b-base_eval_common.sh pubmedqa
bash bash_scripts/eval/qwen3-8b-base_eval_common.sh mathqa_challenge
```

Ejecutar el ejecutable compartido sin un argumento solo imprime los nombres de conjuntos de datos disponibles. Las salidas se escriben en:

```bash
eval_output/main_eval/qwen3_8b_base_<dataset>/
```

Cada directorio de salida contiene:

- `<dataset>_generated.parquet`: respuestas generadas.
- `<dataset>_main_eval.log`: registro final de recomp/precisión impreso por `main_eval`.

Para una prueba inicial, edita el script objetivo o el ejecutable compartido y establece:

```bash
MAX_SAMPLES=2
```

Luego ejecuta el script del conjunto de datos. La evaluación completa usa `MAX_SAMPLES=0`.

## Notas Prácticas

- Para Step-TreeRL, prefiere prompts que fuerzen la segmentación explícita `<step>...</step>`; de lo contrario, la extracción de ramas y el PRM de formato se volverán ruidosos.
- Para GRPO y Step-TreeRL, `actor_rollout_ref.rollout.n` controla las muestras iniciales por prompt. Step-TreeRL también tiene `m`; mantén `m` alineado con `rollout.n` a menos que lo sobrescribas intencionalmente.
- Para trazas largas o variables, prefiere el lote dinámico: `actor_rollout_ref.actor.use_dynamic_bsz=True` y ajusta `ppo_max_token_len_per_gpu`.
- Para Step-TreeRL, establece `actor_rollout_ref.rollout.max_model_len >= data.max_prompt_length + trainer.step_treerl_config.max_token_num`.
- No comitees claves API reales. Usa interpolación de entorno como `${oc.env:MINIMAX_API_KEY}` en scripts de lanzamiento.
