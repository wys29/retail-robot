import argparse
import sys
import json

from utils import IO
from utils import Intent


def main():
    parser = argparse.ArgumentParser(
        description='处理查询和会话ID的脚本',
        epilog='示例: python FAQ_main.py "你好 你好" 会话id'
    )
    
    parser.add_argument('query', type=str, help='用户问题（必需）')
    
    parser.add_argument('sid', type=str, help='会话ID（必需）')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        print("错误：请提供必要的参数")
        parser.print_help()
        sys.exit(1)
    

    
    print(f"处理查询: {args.query}")
    print(f"会话ID: {args.sid}")

    #存储和多轮对话读取
    IO.append_to_dataset(args.query,args.sid)
    querys = IO.get_querys_by_sid(args.sid)
    history_responsed = IO.get_responses_by_sid(args.sid)

    print(f"多轮会话读取: {querys}")

    #实体识别幻觉干预
    alternative_products = IO.recall_alternative_products("。".join(querys) + "。")

    #读取购物车内的实体
    shopping_cart_products = IO.get_shopping_products()

    print("实体识别范围(包含购物车):",alternative_products+shopping_cart_products)

    
    #意图识别+实体识别
    response = Intent.intent_classify(querys,alternative_products,shopping_cart_products)
    
    response = json.loads(response)

    intent = response['intent']
    products = response['products']



    if intent == 1: 
        response = Intent.where_product(querys,products)
    elif intent == 2:
        response = Intent.what_categories(querys,products)
    elif intent == 3:
        response = Intent.want_shopping_tips(querys,products)
    elif intent == 4:
        response = "{\"Answer\":\"抱歉，我还没有学会回答这一类问题。\"}"
    

    print("大模型响应:", response)
    response = json.loads(response)
    
    #大模型应答保存
    IO.write_response_to_dataset(response['Answer'])


if __name__ == "__main__":
    main()