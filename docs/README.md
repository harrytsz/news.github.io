# 人工智能新闻推荐系统
> [!TIP]
> 本项目采用了前后端分离式架构设计，融合了游戏引擎，人脸识别，语音识别，语音合成，智能语音聊天，个性化推荐算法于一体的多模态新闻推荐系统。

## 系统总览

### 登录

当用户输入新闻推荐系统网址后，首先需要完成登录操作。这里默认使用传统的用户名密码登录方式进入系统。与此同时，用户还可以选择人脸识别的方式进行登录。

#### 用户名密码登录
![](https://pic.imgdb.cn/item/613cb6f944eaada739a1bf03.jpg)

#### [人脸识别登录](face/face?id=人脸识别)

当用户选择人脸识别登录时，浏览器会开启摄像头进行人脸捕捉与识别。完成人脸定位与识别之后，系统会匹配人脸数据库中相似度超过阈值的用户进行身份认证以完成登录操作。

![](https://pic.imgdb.cn/item/61443bad2ab3f51d91b03f7a.jpg)

**引导页**

人脸识别通过后，进入网站引导页面：

![](https://pic.imgdb.cn/item/61442eca2ab3f51d919d1af1.jpg)


我将个人简历做成了网页的形式放在“关于我”的部分，点击“关于我”即可跳转至开发者简绍页面。

**“关于我”部分**

导航菜单页面是使用 Cocos Creator 游戏开发引擎制作的具有动画效果和交互操作的页面，开发的目的是为了增强新闻系统的前端 UI 显示效果，毕竟绝大部分用户都是“颜值控”。如果一个系统的 UI 设计的不够美观，会极大地削弱推荐算法的效果。未来我想尝试着去做一个沉浸式的信息流平台，把玩游戏的体验带到信息流系统中来。

![](https://imgstore.harrytsz.com/chrome-capture-2026-01-19-21.gif)


主要的技术分享平台

![](https://img.harrytsz.com/blog202203231523288.gif)
![](https://imgstore.harrytsz.com/chrome-capture-2026-01-19-22.gif)

开发者个人技能

![](https://img.harrytsz.com/blog202203231524486.gif)
![](https://imgstore.harrytsz.com/chrome-capture-2026-01-19-23.gif)

退出“关于我”的页面后，选择进入新闻推荐系统。

![](https://img.harrytsz.com/blog202203231557744.gif)
![](https://imgstore.harrytsz.com/chrome-capture-2026-01-19-24.gif)

#### 新闻网站主页
通过人脸识别或者用户名密码任意方式登陆成功后，系统会自动识别用户身份信息，新闻网站主页“为你推荐”模块会根据用户历史浏览记录推荐给用户可能感兴趣的新闻。

![](https://pic.imgdb.cn/item/61443c992ab3f51d91b18ae8.jpg)

点击新闻候选框或者热度榜即可进入新闻详情页面。

![](https://pic.imgdb.cn/item/61443d322ab3f51d91b26a76.jpg)

新闻页面中的“语音播报”按钮可以开启新闻内容语音播报功能，此功能针对特殊需求的用户群体设计。目前，手势识别技术逐渐成熟，为了增加交互的便捷性，迭代版本中考虑融合“手势识别”模块，可以自定义手势用以触发对应的功能。

目前已经完成了手势识别功能的开发，可以通过手势控制新闻页面的跳转。


### Django 
在后端框架选型时我选择了 Django，主要考虑了开发成本。数据处理部分是基于 Python 技术栈的，选择 Django 避免了多种技术栈来回切换的成本。其他同样优秀的后端框架还有 SpringBoot，当然还有一些轻量级的框架如: Flask、Jetty等框架。如果考虑高并发场景的话，SpringBoot 以及微服务架构可能更加合适。

![](https://pic.imgdb.cn/item/613c80b044eaada739f51aae.jpg)

#### Django后台管理

Django 框架强大的管理能力可以极大降低开发难度，Django Admin 提供了系统数据库的可视化管理与操作页面。提供了针对数据库的“增删改查”操作，非常便捷。

![](https://pic.imgdb.cn/item/6142ca272ab3f51d91d6cb23.jpg)


## 数据存储部分

> [!TIP]
> 数据存储模块的设计思路是：
> 分级存储，把越频繁访问的数据放到越快的数据库甚至缓存中，把海量的全量数据放到廉价且查询速度相对慢的数据库中。


### MySQL
MySQL 是一个强一致性的关系型数据库，一般用来存储比较关键的要求强一致性的信息，比如用户注册信息、用户权限、系统配置等信息。这类信息一般有服务器进行阶段性拉取，或者利用分级缓存进行阶段性的更新，避免因为过于频繁的访问压垮 MySQL。

![](https://pic.imgdb.cn/item/613c80ef44eaada739f57e14.jpg)

#### MySQL中的字段
![](https://pic.imgdb.cn/item/613cb8dc44eaada739aa8e5c.jpg)

### 缓存部分
![](https://pic.imgdb.cn/item/613c812944eaada739f5d8ae.jpg)

用户 Embedding、新闻 Embedding 、场景 Embedding 等这些特征都是在离线环境下训练得到的，推荐服务器是在线上环境中运行的，这些离线特征数据正是通过 Redis 等数据库导入到线上环境中提供给推荐服务器使用。

```python
import redis
pool = redis.ConnectionPool(host='127.0.0.1', port=6379, password=123456)
redisClient = redis.Redis(connection_pool=pool)

def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(5).setNumIterations(10)
    model = word2vec.fit(samples)
    synonyms = model.findSynonyms("158", 20)
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)
    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if saveToRedis:
        for newsId in model.getVectors.keys:
            redisClient.set(redisKeyPrefix + ":" + newsId, model.getVectors(newsId).mkString(" "))
    redisClient.close()
    embeddingLSH(spark, model.getVectors())
    return model
```

## 人工智能模块

### [新闻推荐](rec/rec?=推荐模块)
![](https://pic.imgdb.cn/item/61443fa92ab3f51d91b66cfe.jpg)

### [人脸识别](face/face?id=人脸识别)
![](https://pic.imgdb.cn/item/61443eb22ab3f51d91b50a5b.jpg)

### [语音模块](speech/speech?id=语音处理模块)
![](https://pic.imgdb.cn/item/614461872ab3f51d91e7625b.jpg)

### [智能聊天](speech/speech?id=_2-搭建-wukong-robot-运行环境)
