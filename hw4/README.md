项目结构：
```
| requirements.txt
| README.md
|—— crawler
|   |—— downloads                                   ————————存储爬取的文档
|   | crawler.py                                    ————————爬虫程序，以及对爬取的网页进行一些处理，
|   | title_url_anchor_body.csv                     ————————存储爬取的网页，文件头为 title,url,anchor_text,body 
|   | pretreat.py                                   ————————为爬取的网页增加列号，作为 docID
|   | linenumber_title_url_anchor_body.csv          ————————增加了列号的文件，文件头为 linenum,title,url,anchor_text,body 
|   |
|—— indexer
|   |—— file_inverted_index_chunks                  ————————保存构建文件索引的倒排索引的JSON文件的文件夹
|   |—— file_tf_idf_chunks                          ————————保存构建文件索引的TF-IDF的JSON文件的文件夹
|   |—— inverted_index_chunks                       ————————保存构建全文索引的倒排索引的JSON文件的文件夹
|   |—— tf_idf_chunks                               ————————保存构建全文索引的TF-IDF的JSON文件的文件夹
|   |—— title_inverted_index_chunks                 ————————保存构建标题索引的倒排索引的JSON文件的文件夹
|   |—— title_tf_idf_chunks                         ————————保存构建标题索引的TF-IDF的JSON文件的文件夹
|   | baidu_stopwords.txt                           ————————停用词表
|   | cn_stopwords.txt                              ————————停用词表
|   | title_word_count.csv                          ————————保存构建标题索引的每个url的单词总数
|   | file_word_count.csv                           ————————保存构建文档索引的每个url的单词总数
|   | word_count.csv                                ————————保存构建全文索引的每个url的单词总数
|   | index.py                                      ————————用于构建倒排索引
|   | tokens_cal.py                                 ————————计算每个url的单词总数
|   | tf_idf_cal.py                                 ————————计算每个文档每个单词的 TF-IDF 值
|   |
|—— pageranke
|   | pagerank_analysis.py                          ————————计算每个文档的pagerank分数
|   | pagerank_results.csv                          ————————保存每个文档的pagerank分数，文件头为 url,pagerank
|   |
|—— search  
|   | search.py                                     ————————查询主文件，实现了所有查询服务、网页快照、个性化查询等功能
|   | term_association_search.py                    ————————实现了个性化推荐
|   | query_log.txt                                 ————————历史记录文件，用于保存每次查询返回的前5条记录
|   | result.txt                                    ————————保存每次查询结果的文件
|   |—— page_photos                                 ————————保存网页快照的文件夹
```
需要注意的的是，所有的 python 的文件最好能够在程序所在目录中执行，比如：`crawler.py` 在crawler目录中执行，在以避免执行时，相对路径报错。

同时在执行`index.py`构建索引和执行`tokens_cal.py`计算每个文档的单词总数以及执行`tf_idf_cal.p`y计算 TF-IDF 值，对于不同的索引（全文索引、标题索引以及文档索引），代码中一些文档路径需要修改。

在计算完 TF-IDF 值和 pagerank 分数后即可执行 `search.py`程序来进行查询。