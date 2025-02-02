import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import TavilySearchAPIRetriever
from operator import itemgetter
from dotenv import load_dotenv


# env ファイルの読み込み
load_dotenv()

# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

# Not LCEL
def check_not_lcel():

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザが入力したレシピを教えてください。"),
            ("human", "{dish}")
        ]
    )

    st.write("ユーザが入力したレシピを教えてください。カレー") 

    prompt_value = prompt.invoke({"dish":"カレー"})
    ai_message = model.invoke(prompt_value)
    output = output_parser.invoke(ai_message)
    st.write(f"Not LCEL：{output}")


# LCEL
def check_lcel():

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザが入力したレシピを教えてください。"),
            ("human", "{dish}")
        ]
    )

    st.write("ユーザが入力したレシピを教えてください。カレー") 

    chain = prompt | model | output_parser
    output = chain.invoke({"dish":"カレー"})
    st.write(f"LCEL：{output}")


# stream 
def check_stream():

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザが入力したレシピを教えてください。"),
            ("human", "{dish}")
        ]
    )

    st.write("ユーザが入力したレシピを教えてください。カレー") 
    chunks = ""

    chain = prompt | model | output_parser
    for chunk in chain.stream({"dish":"カレー"}):
        if chunk == "\n":
            st.write(chunks)
            chunks = ""
        else:
            chunks += chunk
    if chunks:
        st.write(chunks)


# batch
def check_batch():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザが入力したレシピを教えてください。"),
            ("human", "{dish}")
        ]
    )

    st.write("ユーザが入力したレシピを教えてください。カレー") 

    chain = prompt | model | output_parser
    output = chain.batch([{"dish":"カレー"},{"dish":"うどん"}])
    st.write(f"batch：{output}")


# chain to chain
def check_chain_to_chain():
    cot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザの質問にステップバイステップで答えてください。"),
            ("human", "{question}")
        ]
    )

    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ステップバイステップで考えた回答から結論だけ抽出してください。"),
            ("human", "{text}")
        ]
    )

    st.write("質問：10 + 2 * 3") 

    cot_chain = cot_prompt | model | output_parser
    summarize_chain = summarize_prompt | model | output_parser

    cot_summarize_chain = cot_chain | summarize_chain

    output = cot_summarize_chain.invoke({"question":"10 + 2 * 3"})

    st.write(f"２つのChainを繋いだ結果  question：10 + 2 * 3 ：{output}")


# RunnableLambda 関数をRunnableにする
def check_runnable_lambda():

    runnalbe_lambda_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたのアシスタントが質問に答えます。"),
            ("human", "{input}")
        ]
    )

    def upper(text: str) -> str:
        return text.upper()
    
    st.write("質問：hello world")
    st.write("RunnableLambda：hello world を大文字に変換する")

    chain = runnalbe_lambda_prompt | model | output_parser | RunnableLambda(upper)
    chain = runnalbe_lambda_prompt | model | output_parser | upper
    
    output = chain.invoke({"input":"hello world"})
    st.write(f"RunnableLambda：{output}")

# check runnable parallel
def check_runnable_parallel(sumrize_flg: bool = False):
    optimistic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは楽観主義者です。ユーザの入力に対して楽観的な意見をください。"),
            ("human", "{topic}")
        ]
    )

    pessimistic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは悲観主義者です。ユーザの入力に対して悲観的な意見をください。"),
            ("human", "{topic}")
        ]
    )

    topic_str = "明日の天気は晴れるかな？"

    optimistic_chain = optimistic_prompt | model | output_parser
    optimistic_chain_out = optimistic_chain.invoke({"topic":topic_str})    
    st.write(f"楽観主義者：{optimistic_chain_out}")

    pessimistic_chain = pessimistic_prompt | model | output_parser
    pessimistic_chain_out = pessimistic_chain.invoke({"topic":topic_str})
    st.write(f"悲観主義者：{pessimistic_chain_out}")

    if sumrize_flg:
        st.write("2つのChainを並列に実行して意見を出して、結果をまとめる")
        summarize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたは客観的なAIです。楽観的な意見と悲観的な意見を1つにまとめてください。"),
                ("human", "楽観的意見:{optimistic_option}\n悲観的意見:{pessimistic_option}")
            ]
        )

        summarize_chain = (
            RunnableParallel(
                {
                    "optimistic_option":optimistic_chain,
                    "pessimistic_option":pessimistic_chain,
                    'topic':itemgetter('topic'),
                }
            ) | summarize_prompt | model | output_parser
        )

        output = summarize_chain.invoke({"topic":topic_str})
        st.write(output)
    else:
        st.write("2つのChainを並列に実行して意見を出すが、結果をまとめない")
        paralell_chain = RunnableParallel(
            {
                "optimistic_option":optimistic_chain,
                "pessimistic_option":pessimistic_chain,
                'topic':itemgetter('topic'),
            }
        )

        output = paralell_chain.invoke({"topic":topic_str})
        st.write(output)


def check_runnable_passthrough(assing_flg: bool = False, astream_flg: bool = False):
    
    question = "東京の今日の天気は"
    prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。
                                              
文脈:"""
{context}
"""
                                              
質問:
{question}
''')
    
    st.write("質問：東京の今日の天気は")

    retriever = TavilySearchAPIRetriever(k=3)

    # RunnablePassthroughの場合
    if assing_flg:
        chain = ( 
                    {
                        "context": retriever, 
                        "question": RunnablePassthrough()
                    } | RunnablePassthrough.assign(answer= prompt | model | StrOutputParser() ) 
                )

        output = chain.invoke(question)
        st.write(f"回答のみ返す：{output}")
    # RunnablePassthroughの場合
    else:
        chain = ( {
                    "context": retriever, 
                    "question": RunnablePassthrough()
                } | prompt | model | StrOutputParser() 
            )

        # if astream_flg:
            # for event in chain.astream_events(question, version="v2"):
            #    print(event, flush=True)

        # else:
        output = chain.invoke(question)
        st.write(f"プロンプトの内容を含めた結果：{output}")

#-------------------------------------
# MAGI ToT Classの実装
#-------------------------------------
# MAGIをLCELで実装する
class MagiSummarize():
    # 初期設定
    def __init__(self, roles):
        self.roles = roles
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.output_parser = StrOutputParser()

    # magiへの問い合わせ処理
    def invoke(self, question):

        st.write(f'=== 以下の質問に答えます ===  \n 質問:{question}  \n {self.roles}の意見を求めます。')

        # 専門家一人ずつに質問する --- (*2a)
        role_chains = self.__create_role_chains(self.roles, "あなたは{role}です。\n \
                                                    ### 指示:質問に真摯に向き合い、\
                                                    {role}らしい意見を述べます。")
        

        # 専門家の答えをまとめた答えを出力する --- (*2b)
        opinion_chain = self.__magi_summarize(question, role_chains)

        # (*2a)と(*2b)を元に専門家のコメントを求める --- (*2c)
        role_chains = self.__create_role_chains(self.roles, 
                                        "あなたは{role}の代表です。\n \
                                            ### 指示:質問に対する答えについて、\
                                            賛否と意見を述べてください。\n \
                                            出力例:\n- 賛成 or 反対: 理由\n")
        # 改めて専門家の意見をまとめる --- (*2d)
        answer_chain = self.__magi_summarize(question, role_chains)

        magi_chain = opinion_chain | answer_chain
        output = magi_chain.invoke({"question":question}, {"callbacks": [ConsoleCallbackHandler()]})

        return output["answer"]

    # 専門家の意見を求める処理
    def __create_role_chains(self, roles, prompt_template):
            role_chains = []
            for role in roles:
                role_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", prompt_template.format(role=role)),
                        ("human", "{question}")
                    ]
                )
                role_chains.append({"question":RunnablePassthrough()} | role_prompt | self.model | self.output_parser)
                role_chain= {"question":RunnablePassthrough()} | role_prompt | self.model | self.output_parser
                role_chain_out = role_chain.invoke({"question":"日本の憲法は何年に制定されましたか？"})
                st.write(f"{role}：{role_chain_out}")
            return role_chains

    # 専門家の意見をまとめる処理
    def __magi_summarize(self, question, role_chains):
        summarize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたは善良で公平な裁判官です。以下の専門家たちの意見を簡潔な結論と理由を提出してください。答えが明確な場合は、そのまま回答してください。"),
                ("human", "元の質問は「{question}」 専門家１の意見:{role1} \n 専門家２の意見:{role2} \ n専門家３の意見:{role3}")
            ]
        )

        summarize_chain = (
            RunnableParallel(
                {
                    "question":itemgetter('question'),
                    "role1":role_chains[0],
                    "role2":role_chains[1],
                    "role3":role_chains[2],
                }
            ) | RunnablePassthrough.assign(answer= summarize_prompt | model |  self.output_parser ) 
        )

        summarize_chain_out = summarize_chain.invoke({"question":question})
        st.write(f"結論：{summarize_chain_out}")

        return summarize_chain


def main(mode):

    # "check_not_lcel", # lcelで記述していないChainを実行
    # "check_lcel", # lcelで記述したChainを実行
    # "check_stream", # streamでChainを実行
    # "check_batch", # batchでChainを実行
    # "check_chain_to_chain", # ChainとChainをつなげる
    # "check_runnable_lambda", # 関数をRunnableにする
    # "check_runnable_parallel", # 2つのChainを並列に実行
    # "check_runnable_parallel_sumrize", # 2つのChainを並列に実行し、結果をまとめる
    # "check_runnable_passthrough", # Questionはそのままの値をプロンプトに渡し、contextには質問の内容をWeb検索制て取得した結果をプロンプトに渡す(回答のみを返す)
    # "check_runnable_passthrough_assing", # Questionはそのままの値をプロンプトに渡し、contextには質問の内容をWeb検索制て取得した結果をプロンプトに渡す(プロンプトも返す)
    # "check_runnable_passthrough_astream", # Questionはそのままの値をプロンプトに渡し、contextには質問の内容をWeb検索制て取得した結果をプロンプトに渡す(Streamで返す)
    # "check_magi", # MAGI ToTを実行
    
    if mode == "check_not_lcel":
        st.write("lcelで記述していないChainを実行")
        check_not_lcel()
    elif mode == "check_lcel":
        st.write("lcelで記述したChainを実行")
        check_lcel()
    elif mode == "check_stream":
        st.write("streamでChainを実行")
        check_stream()
    elif mode == "check_batch":
        st.write("batchでChainを実行")
        check_batch()
    elif mode == "check_chain_to_chain":
        st.write("ChainとChainをつなげる")
        check_chain_to_chain()
    elif mode == "check_runnable_lambda":
        st.write("関数をRunnableにする")
        check_runnable_lambda()
    elif mode == "check_runnable_parallel":
        st.write("2つのChainを並列に実行")
        check_runnable_parallel()
    elif mode == "check_runnable_parallel_sumrize":
        st.write("2つのChainを並列に実行し、結果をまとめる")
        check_runnable_parallel(sumrize_flg=True)
    elif mode == "check_runnable_passthrough":
        st.write("Questionはそのままの値をプロンプトに渡し、contextには質問の内容をWeb検索して取得した結果をプロンプトに渡す(回答のみを返す)")
        check_runnable_passthrough()
    elif mode == "check_runnable_passthrough_assing":
        st.write("Questionはそのままの値をプロンプトに渡し、contextには質問の内容をWeb検索して取得した結果をプロンプトに渡す(プロンプトも返す)")
        check_runnable_passthrough(assing_flg=True)
    elif mode == "check_runnable_passthrough_astream":
        st.write("Questionはそのままの値をプロンプトに渡し、contextには質問の内容をWeb検索して取得した結果をプロンプトに渡す(Streamで返す)")
        check_runnable_passthrough(assing_flg=False, astream_flg=True)
    elif mode == "check_magi":
        st.write("MAGI ToTを実行")
        chekc_magi = MagiSummarize(["医者", "弁護士", "教師"])
        output = chekc_magi.invoke( "日本の憲法は何年に制定されましたか？")
        st.write(f"MAGIの最終的な回答:{output}")

if __name__ == "__main__":

    st.title('LCELの確認')

    mode = st.selectbox(
        '確認する例を選択してください:', 
        [
            "check_not_lcel", 
            "check_lcel", 
            "check_stream", 
            "check_batch", 
            "check_chain_to_chain", 
            "check_runnable_lambda", 
            "check_runnable_parallel", 
            "check_runnable_parallel_sumrize", 
            "check_runnable_passthrough", 
            "check_runnable_passthrough_assing", 
            "check_runnable_passthrough_astream", 
            "check_magi"
        ]
    )

    if st.button('実行'):
        main(mode)



## MAIG への質問例
# 指示 3人の専門家に議論をさせて、質問に対する回答を出してください。
# * IT系スタートアップの社長
# * 総理大臣
# * 五輪金メダリスト
# 質問 日本でITアーキテクトの人口を増やす方法を5つ考えてください。