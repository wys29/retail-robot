import requests
import time
import random
import json
import openai
import os
import pandas as pd

def chat_v2(messages, model="Chatrhino-81B-Pro"):
    """调用大模型进行对话的核心函数"""
    api_keys = ['43510ad4-1456-4b74-8610-81a22becaf86']  # 替换为你的API密钥
    random_api_key = random.choice(api_keys)
    os.environ["OPENAI_API_KEY"] = random_api_key
    os.environ["OPENAI_API_BASE"] = "http://gpt-proxy.jd.com/gateway/azure"
    
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"]
    )
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"{os.environ['OPENAI_API_KEY']}"
    }

    temperature = 0.1
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
            extra_headers=headers,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return json.dumps({"Evidence": "Error", "Answer": "LLM调用失败"})


def generate_messages(user_querys,alternative_product,shopping_cart_products, system_prompt):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"\n用户问题:{user_querys[0]},可选实体:{alternative_product},购物车实体:{shopping_cart_products}" 
        }
    ]
    
    for i in range(1,len(user_querys)):
        messages.append({
        "role": "assistant",
        "content": ''
        })
        

        messages.append({
            "role": "user",
            "content": f"\n用户问题:{user_querys[i]}" 
        })

    return messages


def generate_messages_more_information(user_querys,system_prompt,knowledge,product):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"商品知识:\n{knowledge}\n用户询问的商品名称:{product}\n\n用户问题:{user_querys[0]}" 
        }
    ]
    
    for i in range(1,len(user_querys)):
        messages.append({
        "role": "assistant",
        "content": ''
        })
         
        messages.append({
            "role": "user",
            "content": f"\n用户问题:{user_querys[i]}" 
        })

    return messages


def intent_classify(querys,alternative_products,shopping_cart_products):

        
    system_prompt = '''
        假设你是一位阅读理解专家，当前是京东线下零售超商场景。输入包括用户问题还有可选实体以及购物车内的实体。请根据查询文本精准识别用户查询意图（分类至以下类别）并提取商品实体：但只提取在给定实体范围内的实体（如果范围提供），并按实体在查询中的出现顺序输出。

        意图类别：
        1. 商品寻址类（定位商品位置）
        2. 品类信息类（查询商品有什么品类）
        3. 购物建议类（其他所有购物相关的问题）
        4. 以上都不是

        输出要求：
        - 输出一个JSON对象，格式固定为：{"intent": <intent_category>, "products": [<product_name1>, <product_name2>, ...]}
        - <intent_category> 是整数（1, 2, 3 或 4），表示整体查询意图。
        - 可选实体和购物车实体组成可以识别的实体范围，你不能识别范围以外的实体
        - 实体识别优先级：购物车实体 > 可选实体；当购物车实体不为空的时候，尽量关联至购物车中最匹配的实体。
        - <products> 是一个字符串列表：包含提取的商品实体名称，按在查询中首次出现的顺序排列。如果未提供实体范围，则提取所有提到的商品实体；如果提供实体范围，则只提取在范围内的实体；如果没有可提取实体，列表为空 []。
        - 仅输出JSON对象，禁止输出推理过程、额外文本或格式化文本。
        - 示例参考（输入格式：{"用户问题": "查询文本", "可选实体": [可选实体列表],"购物车实体":[购物车实体]}；输出仅JSON）：
        输入: {"用户问题": "苹果在哪里能买到？", "可选实体": ["苹果", "香蕉"],"购物车实体":[]}
        输出: {"intent": 1, "products": ["苹果"]}

        输入: {"用户问题": "你们有哪些品种的苹果和香蕉？", "可选实体": ["苹果", "牛奶"],"购物车实体":[]}
        输出: {"intent": 2, "products": ["苹果"]}  // 范围含"苹果"、不含"香蕉"，故忽略香蕉，苹果是第一个出现

        输入: {"用户问题": "苹果、橙子和梨都在哪儿？", "可选实体": ["苹果", "橙子", "葡萄"],,"购物车实体":[]}
        输出: {"intent": 1, "products": ["苹果", "橙子"]}  // 梨不在范围，忽略；顺序按出现顺序：苹果第一、橙子第二

        输入: {"用户问题": "健身吃什么牛肉性价比高？", "可选实体": [],,"购物车实体":[]}
        输出: {"intent": 3, "products": ["牛肉"]}  // 未提供范围，提取所有实体

        输入: {"用户问题": "你们店几点开门？", "可选实体": ["苹果"],"购物车实体":[]}
        输出: {"intent": 4, "products": []}  // 无实体提及，空列表

        输入: {"用户问题": "你这有果卖吗", "可选实体": ["水果","苹果","青苹果"],"购物车实体":[]}
        输出: {"intent": 2, "products": ["水果"]}  // 无实体提及，空列表

        '''
    
    messages = generate_messages(querys,alternative_products,shopping_cart_products,system_prompt)

    start_time = time.time()
    response = chat_v2(messages)
    elapsed_time = time.time() - start_time
    
    # print(f"响应时间: {elapsed_time:.2f}秒")
    print("大模型响应:", response)
    return response


def where_product(querys,products):
        
    system_prompt = '''
        角色设定你是一个帮助顾客找到商品位置的专业人工客服。根据提供的商品相关知识，提供给顾客想要商品的具体位置信息。回答要求
        1.以JSON格式回复
        2.如果无法判断用户意图或无法在历史用户问答、商品相关知识中找到相关的问题，直接输出:{"Answer":"暂时无法回答该问题"}2.1.如果是基于你的常识来回答的，也输出为:{"Answer":"暂时无法回答该问题"}
        3.输出的Answer中不能出现对该商品的一些负面评价，比如"确实不好用"、"有用户反馈用一段时间就坏了"、"质量不好"、"容易坏"等。
        4.输出的Answer中不能提及你是根据知识库信息而得到的答案。
        #输出格式{"Answer":"结合找到的知识库信息进行润色并以客服的语气，简明扼要地回复用户"}其中，Answer表示针对用户问题的回复
        #注意事项1.相关知识中，一条json数据中，product表示用户询问的商品，表示回复内容，每对问答用\n分隔，answers中可能与用户问题无关，注意甄别。
        '''

    
    def find_sales_region(products):

        df = pd.read_excel('./dataset/知识库.xlsx', sheet_name='Sheet1')
    
        result = []
        for item in products:
            # 搜索匹配的行（三列中任意一列匹配）
            mask = (df['一级品类'] == item) | (df['二级品类'] == item) | (df['三级品类'] == item)
            matched_row = df.loc[mask].iloc[0] if any(mask) else None
            
            if matched_row is not None:
                region = matched_row['售卖区域']
                # 构建JSON格式字典
                item_data = {"实体": item, "售卖区域": str(region)}
                result.append(json.dumps(item_data, ensure_ascii=False))
        
        return result
            

    knowledge = find_sales_region(products)

    messages = generate_messages_more_information(querys,system_prompt,knowledge,products)

    start_time = time.time()
    response = chat_v2(messages)
    elapsed_time = time.time() - start_time
    
    # print(f"响应时间: {elapsed_time:.2f}秒")
    return response


def what_categories(querys,products): 
    

    system_prompt = '''
        角色设定你是一个帮助顾客给顾客介绍商品品类的专业人工客服。根据提供的商品相关知识，提供给顾客想要商品的具体位置信息。回答要求
        1.以JSON格式回复
        2.如果无法判断用户意图或无法在历史用户问答、商品相关知识中找到相关的问题，直接输出:{"Answer":"暂时无法回答该问题"}2.1.如果是基于你的常识来回答的，也输出为:{"Answer":"暂时无法回答该问题"}
        3.输出的Answer中不能出现对该商品的一些负面评价，比如"确实不好用"、"有用户反馈用一段时间就坏了"、"质量不好"、"容易坏"等。
        4.输出的Answer中不能提及你是根据知识库信息而得到的答案。
        #输出格式{"Answer":"结合找到的知识库信息进行润色并以客服的语气，简明扼要地回复用户"}其中，Answer表示针对用户问题的回复
        #注意事项1.相关知识中，一条json数据中，product表示用户询问的商品，表示回复内容，每对问答用\n分隔，answers中可能与用户问题无关，注意甄别。
        '''
    
    def product_to_json(products):
        
        df = pd.read_excel('./dataset/知识库.xlsx', sheet_name='Sheet1')
    
        result_list = []
        for product in products:
            # 检查三级品类匹配
            if product in df['三级品类'].values:
                result_list.append({"实体": product, "下级实体": "无下级品类"})
                continue
            
            # 检查二级品类匹配
            elif product in df['二级品类'].values:
                # 获取该二级品类对应的所有三级品类
                lower_entities = df[df['二级品类'] == product]['三级品类'].dropna().unique().tolist()
            
            # 检查一级品类匹配
            elif product in df['一级品类'].values:
                # 获取该一级品类对应的所有二级品类
                lower_entities = df[df['一级品类'] == product]['二级品类'].dropna().unique().tolist()
            
            # 未找到匹配的情况
            else:
                lower_entities = []
            
            # 构建结果字典
            result_list.append({
                "实体": product,
                "下级实体": lower_entities
            })
        
        return json.dumps(result_list, ensure_ascii=False)
        

    knowledge = product_to_json(products)
    
    messages = generate_messages_more_information(querys,system_prompt,knowledge,products)

    start_time = time.time()
    response = chat_v2(messages)
    elapsed_time = time.time() - start_time
    
    # print(f"响应时间: {elapsed_time:.2f}秒")
    
    return response



def want_shopping_tips(querys,product):

    system_prompt = '''
        角色设定你是一个帮助顾客给顾客提供购物建议的专业人工客服。根据提供的商品相关知识，提供给顾客想要商品的具体位置信息。回答要求
        1.以JSON格式回复
        2.如果无法判断用户意图或无法在历史用户问答、商品相关知识中找到相关的问题，直接输出:{"Answer":"暂时无法回答该问题"}2.1.如果是基于你的常识来回答的，也输出为:{"Answer":"暂时无法回答该问题"}
        3.输出的Answer中不能出现对该商品的一些负面评价，比如"确实不好用"、"有用户反馈用一段时间就坏了"、"质量不好"、"容易坏"等。
        4.输出的Answer中不能提及你是根据知识库信息而得到的答案。
        #输出格式{"Answer":"结合找到的知识库信息进行润色并以客服的语气，简明扼要地回复用户"}其中，Answer表示针对用户问题的回复
        #注意事项1.相关知识中，一条json数据中，product表示用户询问的商品，表示回复内容，每对问答用\n分隔，answers中可能与用户问题无关，注意甄别。
        '''
    
    def product_to_json(products):
        
        """
        获取商品知识库信息
        :param goods_list: 商品列表，如["苹果", "橙子"]
        :return: 知识列表，格式如[苹果知识, 橙子知识]，每个元素是包含所有匹配行的JSON字符串列表
        """
        # 读取Excel文件
        try:
            df = pd.read_excel('./dataset/知识库.xlsx', sheet_name='Sheet1')
            # 处理空值
            df = df.fillna('')
        except Exception as e:
            print(f"文件读取失败: {e}")
            return []
        
        result = []
        blacklist_columns = ['售卖区域']
        
        for goods in products:
            # 查找匹配行：商品出现在任意品类列中
            matched_rows = df[
                (df['一级品类'] == goods) |
                (df['二级品类'] == goods) |
                (df['三级品类'] == goods)
            ]

            if blacklist_columns:
                valid_columns = [col for col in df.columns 
                                if col not in [c.strip() for c in blacklist_columns]]
                matched_rows = matched_rows[valid_columns]
                        
            # 将每行转换为字典格式
            goods_knowledge = []
            for _, row in matched_rows.iterrows():
                # 获取整行数据并转换为字典
                row_dict = row.to_dict()
                # 转换为JSON字符串
                goods_knowledge.append(json.dumps(row_dict, ensure_ascii=False))
            
            result.append(goods_knowledge)
        
        return result
        

    knowledge = product_to_json(product)
    
    messages = generate_messages_more_information(querys,system_prompt,knowledge,product)

    start_time = time.time()
    response = chat_v2(messages)
    elapsed_time = time.time() - start_time
    
    # print(f"响应时间: {elapsed_time:.2f}秒")
    return response