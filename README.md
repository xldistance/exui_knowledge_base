使用exui推荐使用GPTQ,EXL2模型,知识库依赖langchain-chatchat的知识库需要放在chatchat同目录下运行，使用exui_knowledge_base之前需要在chatchat正确配置连接ElasticSearch向量数据库，然后在chatchat上传文件到ES数据库之后才可以在exui里面使用

使用exui_knowledge_base之前别忘了运行elasticsearch.bat连接ES数据库

用exui_server.py替换掉Langchain-Chatchat\server\knowledge_base\kb_service\es_kb_service.py

更新最新的langchain在site-packages\langchain_community\vectorstores\elasticsearch.py里面添加这两行
![QQ截图20231220132100](https://github.com/xldistance/exui_knowledge_base/assets/29418474/6d784893-2b83-4b37-ae02-8b22b157884c)
