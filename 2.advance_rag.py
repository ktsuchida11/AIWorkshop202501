import streamlit as st
import os
from typing import Any

from pydantic import BaseModel, Field
from langchain_community.document_loaders import GitLoader
from langchain_community.retrievers import TavilySearchAPIRetriever, BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.documents import Document

from langchain_cohere import CohereRerank
from dotenv import load_dotenv

# env ファイルの読み込み
load_dotenv()


# ここは事前にやってある（時間がかかるため）
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

persist_directory = "./chroma_db" # データベースを保存するディレクトリ

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Chromaデータベースを永続化

if not os.path.exists(persist_directory):

    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter,
    )

    git_documents = loader.load()
    print(len(git_documents))

    db = Chroma.from_documents(git_documents, embeddings, persist_directory=persist_directory)
    db.persist()
    print(f"Chromaデータベースを {persist_directory} に保存しました。")
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print(f"既存のChromaデータベースを {persist_directory} からロードしました。")

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。
                                          
文脈："""
{context}
"""

質問：{question}
''')

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever = db.as_retriever()

output_parser = StrOutputParser()


def check_rag(qestion: str) -> str:

    chain = {
        "question": RunnablePassthrough(),
        "context": retriever,
    } | RunnablePassthrough.assign(answer= prompt | model | output_parser)

    output = chain.invoke(qestion) # "Langchainの概要を教えてください。"
    return output

# -------------------------------------------------------
# HydEを使って検索
# 生成AIで質問の回答を作成してRAGの検索に利用して最終的な回答を生成する
#--------------------------------------------------------

# HyDEを使って検索
def check_hyde(qestion: str) -> bool:

    # AIに回答を作成させるためのテンプレート
    hypothetical_prompt = ChatPromptTemplate.from_template("""\
    以下の質問に回答する一文を書いてください。
    質問：{question}
    """)

    # 質問の回答を作成するchainを作成
    hypothetical_chain = hypothetical_prompt | model | output_parser

    # 途中経過確認用
    hypothetical_chain_out = hypothetical_chain.invoke(qestion, {"callbacks": [ConsoleCallbackHandler()]})
    st.write("==== 生成AIで作成した仮の回答を表示 ====")
    st.write(hypothetical_chain_out)
    # 途中経過確認用　ここまで

    # 生成AIで作成した回答を使ってRAGを検索する
    # RAGの検索結果をコンテキストとして利用して質問の回答を生成AIを利用して出力する
    chain = {
        "question": RunnablePassthrough(),
        "context": hypothetical_chain | retriever
    } | RunnablePassthrough.assign(answer= prompt | model | output_parser)

    output = chain.invoke(qestion, {"callbacks": [ConsoleCallbackHandler()]}) # "Langchainの概要を教えてください。"
    return output

# -------------------------------------------------------
# MultiQueryRetrieverを使って検索（自前で作成するものとライブラリを利用するものがある）
# 生成AIで複数の質問の回答を作成してRAGの検索に利用して、複数の検索結果から最終的な回答を生成する
#--------------------------------------------------------
# 複数の質問の回答を得るためのプロンプト
query_generate_prompt = ChatPromptTemplate.from_template("""\
    質問に対してベクターデータベースから関連文書を検索するために、
    ３つの異なる検索クエリを生成してください。
    距離ベースの類似性検索の限界を克服するために
    ユーザの質問いき対して複数の視点を提供することが目標です。
                                                           
    質問：{question}
    """)

#--------------------------------------------------------
# 自前のMultiQueryRetriever
# 質問に対して検索クエリを３つ生成してそれぞれをRAGに問い合わせる
#--------------------------------------------------------
# RAGで検索するためのリストを作成するクラス
class QueryGenarationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリリスト")

# 質問に対して検索クエリを３つ生成してそれぞれをRAGに問い合わせる
# 3つのRAGの検索結果を組み合わせてコンテキストして質問に回答する
def check_multiqueryretriever(question: str) -> str:

    # 生成AIで質問から３つの検索クエリを生成する
    # 3つの検索クエリをリストにまとめる
    query_genrate_chain = query_generate_prompt | model.with_structured_output(QueryGenarationOutput) | RunnableLambda(lambda x: x.queries)

    # 途中経過確認用
    query_genrate_chain_out = query_genrate_chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]})
    st.write("==== MultiQueryRetrieverの検索結果を表示 ====")
    st.write(query_genrate_chain_out)
    # 途中経過確認用　ここまで

    # リストで受け取った検索クエリをRAGに問い合わせる
    # RAGの検索結果をリストで受け取る
    # RAGの検索結果をコンテキストとして質問の回答を出す
    chain = {
        "question": RunnablePassthrough(),
        "context": query_genrate_chain | retriever.map(),
    } | RunnablePassthrough.assign(answer= prompt | model | output_parser)

    output = chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]}) # "Langchainの概要を教えてください。"
    return output


#--------------------------------------------------------
# MultiQueryRetriever-Lib
# ユーザが入力したクエリに対してLLMを用いてクエリを複数パターンに拡張する手法
#--------------------------------------------------------
# Textを受け取り、改行で分割してリストにするクラス
class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines

# ライブラリのMultiQueryRetrieverを利用して検索する
def check_multiqueryretriever_lib(question: str) -> str:

    output_list_parser = LineListOutputParser()

    # 生成AIで質問から３つの検索クエリを生成する 
    # 3つの検索クエリをリストにまとめる  
    query_genrate_chain = query_generate_prompt | model | output_list_parser 

    # RAGの検索をまとめて実施する
    multi_retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=query_genrate_chain, parser_key="lines"
    )

    # 途中経過確認用
    multi_retriever_out = multi_retriever.invoke(question, {"callbacks": [ConsoleCallbackHandler()]})
    st.write("==== MultiQueryRetrieverの検索結果を表示 ====")
    st.write(multi_retriever_out)
    # 途中経過確認用　ここまで

    chain = {
        "question": RunnablePassthrough(),
        "context": multi_retriever,
    } | RunnablePassthrough.assign(answer= prompt | model | output_parser)
    # promptは、以下の文脈だけを踏まえて質問に回答してください。文脈：{context}　質問：{question}
    # modelは、ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    output = chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]}) # "Langchainの概要を教えてください。"
    return output

#--------------------------------------------------------


#--------------------------------------------------------
# RAG-Fusion
# MultiQueryRerieverの検索結果をRAG-Fusionを使って絞る
# 複数の検索結果に対してスコアをつけてその上位の結果をコンテキストとして生成AIの質問に利用して回答を生成する
#--------------------------------------------------------

#--------------------------------------------------------
# RAG-Fusion自前
# 自前のMultiQueryRetrieverとRAG-Fusionを使って検索を行う
#--------------------------------------------------------
#--------------------------------------------------------
# 自前のMultiQueryRetrieverとRAG-Fusionを使って検索を行う
def check_rag_fuision(question: str) -> str:

    # 自前のMultiQueryRetrieverの出力を解析する
    query_genrate_chain = query_generate_prompt | model.with_structured_output(QueryGenarationOutput) | RunnableLambda(lambda x: x.queries)

    # 途中経過確認用
    query_genrate_chain_out = query_genrate_chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]})
    st.write("MultiQueryRetrieverの検索結果を表示")
    st.write(query_genrate_chain_out)
    # 途中経過確認用　ここまで

    # RAGの検索をまとめて実施する
    rag_fusion_chain = {
        "question": RunnablePassthrough(),
        "context":  query_genrate_chain | retriever.map() | reciprocal_rag_fusion,
    } | RunnablePassthrough.assign(answer= prompt | model | output_parser)

    # promptは、以下の文脈だけを踏まえて質問に回答してください。文脈：{context}　質問：{question}
    # modelは、ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # retrieverは、Chromaの検索結果

    output = rag_fusion_chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]}) # "Langchainの概要を教えてください。"

    return output

# Common 検索結果を並び変えてスコアを計算する処理
def reciprocal_rag_fusion(retriever_outputs: list[list[Document]], k: int = 60,) -> list[str]:

    # 各ドキュメントのコンテンツ（文字列）とそのスコアの対応を保持する辞書を準備
    content_score_mapping = {}

    # 検索クエリごとにループ
    # 検索結果を並び変えてスコアを計算する処理
    for docs in retriever_outputs:
        # ドキュメントごとにループ
        for rank, doc in enumerate(docs):
            content = doc.page_content

            # 初めて登場したコンテンツの場合はスコアを０で初期化
            if content not in content_score_mapping:
                content_score_mapping[content] = 0

            # ( 1/ (rank + 1) ) をスコアに加算
            content_score_mapping[content] += 1 / (rank + 1)

    # スコアの高い順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]

#--------------------------------------------------------
# RAG-Fusion-Lib
# ライブラリのMultiQueryRetrieverとRAG-Fusionを使って検索を行う
#--------------------------------------------------------
# -------------------------------------------------------
# RAG-Fusion-Libを使って検索
def check_rag_fuision_lib(question: str) -> str:

    output_list_parser = LineListOutputParser()

    # MultiQueryRetrieverの出力を解析する
    query_genrate_chain = query_generate_prompt | model | output_list_parser

    # RAGの検索をまとめて実施する
    multi_retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=query_genrate_chain, parser_key="lines"
    )

    # 途中経過確認用
    multi_retriever_out = multi_retriever.invoke(question, {"callbacks": [ConsoleCallbackHandler()]})
    st.write("MultiQueryRetrieverの検索結果を表示")
    st.write(multi_retriever_out)
    # 途中経過確認用　ここまで

    rag_fusion_chain = {
        "question": RunnablePassthrough(),
        "context": multi_retriever | reciprocal_rag_fusion,
    }  | RunnablePassthrough.assign(answer= prompt | model | output_parser)

    output = rag_fusion_chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]}) # "Langchainの概要を教えてください。"

    return output

#--------------------------------------------------------

#--------------------------------------------------------
# Rerank
# RAGの検索結果をRerankを使って並びかえて最終的な回答を生成する
# 質問から普通に検索を行い、その結果をRerankを使って並びかえて最終的な回答を生成する
#--------------------------------------------------------
#--------------------------------------------------------
# Rerankを使って検索
def check_rag_rerank(question: str) -> str:

    # RAGの検索結果確認用
    st.write("==== RAGの検索結果を表示 ===")
    st.write(check_rag(question))
    # RAGの検索結果確認用　ここまで

    # 途中経過確認用
    rerank_chain_out = {
            "question": RunnablePassthrough(),
            "documents": retriever
        } | RunnablePassthrough.assign(context=rerank)
    st.write("==== RAGの検索結果をrerankした結果を表示 ===")
    st.write(rerank_chain_out.invoke(question) )
    # 途中経過確認用　ここまで

    rerank_chain = {
            "question": RunnablePassthrough(),
            "documents": retriever
        } | RunnablePassthrough.assign(context=rerank) | RunnablePassthrough.assign(answer= prompt | model | output_parser)
    # promptは、以下の文脈だけを踏まえて質問に回答してください。文脈：{context}　質問：{question}
    # modelは、ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # retrieverは、Chromaの検索結果

    output = rerank_chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]}) # "Langchainの概要を教えてください。"

    return output

# 10件の検索結果をRerankする
def rerank(inp: dict[str, Any], top_n: int = 10) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    # CohereのRerankモデルを使っている
    # rerank-multilingual-v3.0　は、多言語対応のRerankモデル　有料サービスだがお試し利用もできる
    # cohereのサイト：https://cohere.com/
    # AWS Bedrockでも使える：https://aws.amazon.com/jp/bedrock/cohere/
    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)

    return cohere_reranker.compress_documents(documents=documents, query=question)

#--------------------------------------------------------
# Hybrid Retirever
# 質問をChromaとBM25で検索を行いそれぞれの検索結果を得る
# ２つ検索結果をRAG-Fusionでスコアをつけて抽出して最終的に１つ回答を出力する
#--------------------------------------------------------
#--------------------------------------------------------
# 同じドキュメントソースでHybrid Retireverを使って検索
def check_hybrid_retirever_same_source(question: str) -> str:

    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter,
    )

    git_documents = loader.load()

    # chromaの検索結果を取得
    chroma_retriever = retriever.with_config(
        {"run_name": "chroma_retriever"}
    )
    bm25_retriever = BM25Retriever.from_documents(git_documents).with_config(
        {"run_name": "bm25_retriever"}
    )

    hybrid_retirever = (
        RunnableParallel(
            {
                "chroma_documents": chroma_retriever,
                "bm25_documents": bm25_retriever,
            }
        ) 
        | (lambda x: x["chroma_documents"] + x["bm25_documents"])
    )

    # 途中経過確認用
    hybrid_retirever_out = hybrid_retirever.invoke(question, {"callbacks": [ConsoleCallbackHandler()]})
    st.write("==== Hybrid Retirever Same Sourceの検索結果を表示 ====")
    st.write(hybrid_retirever_out)
    # 途中経過確認用　ここまで

    hybrid_retirever_chain = {
        "question": RunnablePassthrough(),
        "context": hybrid_retirever,
    } | RunnablePassthrough.assign(answer= prompt | model | output_parser)
    # promptは、以下の文脈だけを踏まえて質問に回答してください。文脈：{context}　質問：{question}
    # modelは、ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    output = hybrid_retirever_chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]}) # "Langchainの概要を教えてください。"

    return output

#--------------------------------------------------------
# 別のドキュメントソースでHybrid Retireverを使って検索
def check_hybrid_retirever_other_source(question: str, is_hyde_enabled=False) -> str:

    # chromaの検索結果を取得
    chroma_retriever = retriever.with_config(
        {"run_name": "chroma_retriever"}
    )
    # Web検索をベクトルで行うretriever
    # Webの検索結果を取得
    tavily_retriever = TavilySearchAPIRetriever(k=3).with_config(
         {"run_name": "tavily_retriever"}
    )

    if is_hyde_enabled:
        hybrid_retirever = (
            RunnableParallel(
                {
                    "chroma_documents": chroma_retriever,
                    "tavily_documents": tavily_retriever,
                }
            ) 
            | (lambda x: x["chroma_documents"] + x["tavily_documents"])
        )
    else:
        # AIに回答を作成させるためのテンプレート
        hypothetical_prompt = ChatPromptTemplate.from_template("""\
        以下の質問に回答する一文を書いてください。
        質問：{question}
        """)

        # 質問の回答を作成するchainを作成
        hypothetical_chain = hypothetical_prompt | model | output_parser

        hybrid_retirever = (
            RunnableParallel(
                {
                    "chroma_documents": hypothetical_chain | chroma_retriever,
                    "tavily_documents": hypothetical_chain | tavily_retriever,
                }
            ) 
            | (lambda x: x["chroma_documents"] + x["tavily_documents"])
        )

    # 途中経過確認用
    hybrid_retirever_out = hybrid_retirever.invoke(question, {"callbacks": [ConsoleCallbackHandler()]})
    st.write("==== Hybrid Retirever Other Sourceの検索結果を表示 ====")
    st.write(hybrid_retirever_out)
    # 途中経過確認用　ここまで

    hybrid_retirever_chain = {
        "question": RunnablePassthrough(),
        "context": hybrid_retirever,
    } | RunnablePassthrough.assign(answer= prompt | model | output_parser)
    # promptは、以下の文脈だけを踏まえて質問に回答してください。文脈：{context}　質問：{question}
    # modelは、ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    output = hybrid_retirever_chain.invoke(question, {"callbacks": [ConsoleCallbackHandler()]}) # "Langchainの概要を教えてください。"

    return output

def main(model_name: str, question: str) -> str:

    if model_name == "check_rag":
        output = check_rag(question)
        st.write(f"Ragの回答:  \n{output}")
    elif model_name == "check_hyde":
        # 自前のHyDEを使って検索
        output = check_hyde(question)
        st.write(f"自前のHyDEの回答:  \n{output}")
    elif model_name == "check_multiqueryretriever":
        # 自前の Multi Query Retrieverを使って検索
        output = check_multiqueryretriever(question)
        st.write(f"自前のMultiQueryRetrieverの回答: \n{output}")
    elif model_name == "check_multiqueryretriever_lib":
        # MultiQueryRetrieverを使って検索
        output = check_multiqueryretriever_lib(question)
        st.write(f"MultiQueryRetrieverの回答:  \n{output}")
    elif model_name == "check_rag_fuision":
        # 自前の Multi Query Retrieverを使って検索
        output = check_rag_fuision(question)
        st.write(f"自前のMultiQueryRetrieverのRAG-Fusio回答: \n{output}")
    elif model_name == "check_rag_fuision_lib":
        # MultiQueryRetrieverを使って検索
        output = check_rag_fuision_lib(question)
        st.write(f"MultiQueryRetrieverのRAG-Fusion回答:  \n{output}")
    elif model_name == "check_rag_rerank":
        # Rerankを使って検索
        output = check_rag_rerank(question)
        st.write(f"Rerankの回答:  \n{output}")
    elif model_name == "check_hybrid_retirever_same_source":
        # Hybrid Retireverを使って検索
        output = check_hybrid_retirever_same_source(question)
        st.write(f"Hybrid Retireverの回答:  \n{output}")
    elif model_name == "check_hybrid_retirever_other_source":
        # Hybrid Retireverを使って検索
        output = check_hybrid_retirever_other_source(question)
        st.write(f"Hybrid Retireverの回答:  \n{output}")
    elif model_name == "check_hybrid_retirever_other_source_hyde":
        # Hybrid RetireverをHydeも使って検索
        output = check_hybrid_retirever_other_source(question, is_hyde_enabled=True)
        st.write(f"Hybrid Retirever Hydeの回答:  \n{output}")


if __name__ == "__main__":

    st.title("Advance Rag")

    question = st.text_area('質問を入力してください: 例 Langchainの概要を教えてください。')

    mode = st.selectbox(
        '確認する例を選択してください:', 
        [
            "check_rag",
            "check_hyde",
            "check_multiqueryretriever",
            "check_multiqueryretriever_lib",
            "check_rag_fuision",
            "check_rag_fuision_lib",
            "check_rag_rerank",
            "check_hybrid_retirever_same_source",
            "check_hybrid_retirever_other_source",
            "check_hybrid_retirever_other_source_hyde"
        ]
    )

    if st.button('実行'):
        main(mode, question)





    