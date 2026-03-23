

# from hanlp_restful import HanLPClient
# # 支持zh中文，en英语，ja日本语，mul多语种
# HanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='en')
# amr = HanLP.abstract_meaning_representation('The boy wants the girl to believe him.')
# print(amr)


import hanlp
amr_parser = hanlp.load(hanlp.pretrained.amr.AMR3_GRAPH_PRETRAIN_PARSER)
amr = amr_parser('The administrative service area is southwest of the cultural area.')
print(amr)


from amr_logic_converter import AmrLogicConverter

converter = AmrLogicConverter()

AMR = """
(x / boy
    :ARG0-of (e / giggle-01
        :polarity -))
"""

logic = converter.convert(amr)
print(logic)
