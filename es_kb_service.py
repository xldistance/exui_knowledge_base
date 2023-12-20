from typing import List
import os
import shutil
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from configs import KB_ROOT_PATH, EMBEDDING_MODEL, EMBEDDING_DEVICE, VECTOR_SEARCH_TOP_K,SCORE_THRESHOLD
from server.knowledge_base.kb_service.base import KBService, SupportedVSType
from server.utils import load_local_embeddings
from elasticsearch import Elasticsearch,BadRequestError
from configs import logger
from configs import kbs_config

class ESKBService(KBService):

    def do_init(self):
        self.kb_path = self.get_kb_path(self.kb_name)
        self.index_name = kbs_config[self.vs_type()]['index_name']
        self.IP = kbs_config[self.vs_type()]['host']
        self.PORT = kbs_config[self.vs_type()]['port']
        self.user = kbs_config[self.vs_type()].get("user",'')
        self.password = kbs_config[self.vs_type()].get("password",'')
        self.dims_length = kbs_config[self.vs_type()].get("dims_length",None)
        self.embeddings_model = load_local_embeddings(self.embed_model, EMBEDDING_DEVICE)
        try:
            # ES python客户端连接（仅连接）
            if self.user != "" and self.password != "":
                self.es_client_python =  Elasticsearch(f"https://{self.IP}:{self.PORT}",
                basic_auth=(self.user,self.password),verify_certs=False,request_timeout=30)
            else:
                logger.warning("ES未配置用户名和密码")
                self.es_client_python = Elasticsearch(f"https://{self.IP}:{self.PORT}",verify_certs=False,request_timeout=30)
        except ConnectionError:
            logger.error("连接到 Elasticsearch 失败！")
            raise ConnectionError
        except Exception as e:
            logger.error(f"Error 发生 : {e}")
            raise e

        try:
            # langchain ES 连接、创建索引
            if self.user != "" and self.password != "":
                self.db_init = ElasticsearchStore(
                es_url=f"https://{self.IP}:{self.PORT}",
                index_name=self.index_name,
                query_field="context",
                vector_query_field="dense_vector",
                embedding=self.embeddings_model,
                es_user=self.user,
                es_password=self.password,
                strategy=ElasticsearchStore.ExactRetrievalStrategy(),       #精确搜索
                #strategy=ElasticsearchStore.ApproxRetrievalStrategy(),     #近似搜索
            )
            else:
                logger.warning("ES未配置用户名和密码")
                self.db_init = ElasticsearchStore(
                    es_url=f"https://{self.IP}:{self.PORT}",
                    index_name=self.index_name,
                    query_field="context",
                    vector_query_field="dense_vector",
                    embedding=self.embeddings_model,
                    strategy=ElasticsearchStore.ExactRetrievalStrategy()
                )
        except ConnectionError:
            print("### 初始化 Elasticsearch 失败！")
            logger.error("### 初始化 Elasticsearch 失败！")
            raise ConnectionError
        except Exception as e:
            logger.error(f"Error 发生 : {e}")
            raise e
        """try:
            # 首先尝试通过es_client_python创建
            self.es_client_python.indices.create(index=self.index_name)
        except BadRequestError as e:
            logger.error("创建索引失败,重试")
            try:
                # 尝试通过db_init创建索引
                self.db_init._create_index_if_not_exists(
                                                        index_name=self.index_name,
                                                        dims_length=self.dims_length
                                                        )
            except Exception as e:
                logger.error("创建索引失败...")
                logger.error(e)"""
    @staticmethod
    def get_kb_path(knowledge_base_name: str):
        return os.path.join(KB_ROOT_PATH, knowledge_base_name)

    @staticmethod
    def get_vs_path(knowledge_base_name: str):
        return os.path.join(ESKBService.get_kb_path(knowledge_base_name), "vector_store")

    def do_create_kb(self):
        if os.path.exists(self.doc_path):
            if not os.path.exists(os.path.join(self.kb_path, "vector_store")):
                os.makedirs(os.path.join(self.kb_path, "vector_store"))
            else:
                logger.warning("directory `vector_store` already exists.")

    def vs_type(self) -> str:
        return SupportedVSType.ES

    def _load_es(self, docs, embed_model):
        # 将docs写入到ES中
        try:
            # 连接 + 同时写入文档
            if self.user != "" and self.password != "":
                self.db = ElasticsearchStore.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"https://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        es_user=self.user,
                        es_password=self.password
                    )
            else:
                self.db = ElasticsearchStore.from_documents(
                        documents=docs,
                        embedding=embed_model,
                        es_url= f"https://{self.IP}:{self.PORT}",
                        index_name=self.index_name,
                        distance_strategy="COSINE",
                        query_field="context",
                        vector_query_field="dense_vector",
                        )
        except ConnectionError as ce:
            print(ce)
            print("连接到 Elasticsearch 失败！")
            logger.error("连接到 Elasticsearch 失败！")
        except Exception as e:
            logger.error(f"Error 发生 : {e}")
            print(e)



    def do_search(self, query:str, top_k: int=VECTOR_SEARCH_TOP_K, request_timeout: int = 30) ->List[Document]:
        # 文本相似性检索
        docs = self.db_init.similarity_search_with_score(query=query,
                                         k=top_k,request_timeout = request_timeout)
        return docs


    def do_delete_doc(self, kb_file, **kwargs):
        if self.es_client_python.indices.exists(index=self.index_name):
            # 从向量数据库中删除索引(文档名称是Keyword)
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": kb_file.filepath
                    }
                }
            }
            # 注意设置size，默认返回10个。
            search_results = self.es_client_python.search(body=query, size=50)
            delete_list = [hit["_id"] for hit in search_results['hits']['hits']]
            if len(delete_list) == 0:
                return None
            else:
                for doc_id in delete_list:
                    try:
                        self.es_client_python.delete(index=self.index_name,
                                                     id=doc_id,
                                                     refresh=True)
                    except Exception as e:
                        logger.error("ES Docs Delete Error!")

            # self.db_init.delete(ids=delete_list)
            #self.es_client_python.indices.refresh(index=self.index_name)


    def do_add_doc(self, docs: List[Document], **kwargs):
        '''向知识库添加文件'''
        print(f"server.knowledge_base.kb_service.es_kb_service.do_add_doc 输入的docs参数长度为:{len(docs)}")
        print("*"*100)
        self._load_es(docs=docs, embed_model=self.embeddings_model)
        # 获取 id 和 source , 格式：[{"id": str, "metadata": dict}, ...]
        print("写入数据成功.")
        print("*"*100)
        
        if self.es_client_python.indices.exists(index=self.index_name):
            file_path = docs[0].metadata.get("source")
            query = {
                "query": {
                    "term": {
                        "metadata.source.keyword": file_path
                    }
                }
            }
            search_results = self.es_client_python.search(body=query)
            if len(search_results["hits"]["hits"]) == 0:
                raise ValueError("召回元素个数为0")
        info_docs = [{"id":hit["_id"], "metadata": hit["_source"]["metadata"]} for hit in search_results["hits"]["hits"]]
        return info_docs


    def do_clear_vs(self):
        """从知识库删除全部向量"""
        if self.es_client_python.indices.exists(index=self.kb_name):
            self.es_client_python.indices.delete(index=self.kb_name)


    def do_drop_kb(self):
        """删除知识库"""
        # self.kb_file: 知识库路径
        if os.path.exists(self.kb_path):
            shutil.rmtree(self.kb_path)