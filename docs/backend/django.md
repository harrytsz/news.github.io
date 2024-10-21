## Django 后端开发

这两套技术方案具有各自不同的优缺点，其中 Django 提供了后端一站式解决方案，Django 强大的 ORM 对象关系映射技术可以帮助开发者轻松管理和操纵数据，ORM 建立模型类和表之间的对应关系，允许我们通过面向对象的方式来操作数据库，对数据库的操作都转化成对类属性和方法的操作，避免通过 SQL 语句来操作数据库。实现了数据模型与数据库的解耦，屏蔽了不同数据库操作上的差异。

ORM 框架的缺点也很明显，对于较为复杂的业务，使用成本较高。根据对象的操作转换成 SQL 语句，根据查询的结果转化成对象，在映射过程中有性能损失。

![](https://pic.imgdb.cn/item/61cec6412ab3f51d91b4f24f.jpg)

### [后台管理页面](backend/backend?id=Django后台管理)
![](https://pic.imgdb.cn/item/6142ca272ab3f51d91d6cb23.jpg)

### 用户登录

#### 用户名密码登录
```python
# 选择用户登录
@csrf_exempt
def login(request):
    if request.method == "GET":
        result = dict()
        result["users"]=ALLOW_USERS
        result["tags"]=ALLOW_TAGS
        return JsonResponse(result)
    elif request.method == "POST":
        # 从前端获取用户名 并写入 session
        uname = request.POST.get('username')
        request.session["username"]=uname
        # 前端将标签以逗号拼接的字符串形式返回
        tags= request.POST.get('tags')
        return JsonResponse({"username": uname, "tags": tags,"baseclick":0 , "code": 1})
```

### 后端提供的API
```js
// 主页分类数据
export const getCateNewsData = (getdata) => fetch('/api/index/home/', getdata, 'get')
// 主页分类
export const getCateData = () => fetch('/api/news/cates/', '', 'get')
// 获取新闻详情
export const getNewsData = (newsInfo) => fetch('/api/news/one/', newsInfo, 'get')
// 获取用户以及标签
export const getLogin = () => fetch('/api/index/login/', '', 'get')
// 登录
export const login = (loginInfo) => fetch('/api/index/login/', loginInfo, 'post')
// 退出切换用户
export const layout = () => fetch('/api/index/switchuser/', '', 'get')

//////////////////////////////////////////

// facelogin-bak - index
export const faceHome = () => fetch('/api/facelogin-bak/', '', 'get')
// facelogin-bak -
export const faceRecognition = () => fetch('/api/face_recognition/')

```

### 新闻主页 API 实现

```python
# 主页
def home(request):
    # 从前端请求中获取cate
    _cate = request.GET.get("cateid")
    if "username" not in request.session.keys():
        return JsonResponse({ "code":0 })
    total = 0 # 总页数
    # 如果cate 是为你推荐，走该部分逻辑 tag_flag = 0 表示不是从标签召回数据
    if _cate == "1":
        news, news_hot_value = getRecNews(request)
    # 如果cate 是热度榜，走该部分逻辑
    elif _cate == "2":
        news,news_hot_value = getHotNews()
    # 其他正常的请求获取
    else:
        _page_id = int(request.GET.get("pageid"))
        news = new.objects.filter(new_cate=_cate).order_by("-new_time")
        total = news.__len__()
#         news = new[_page_id * 10:(_page_id+1) * 10]
        news = news[(_page_id-1) * 10:_page_id * 10]  # 解决了不显示近期新闻的 bug
    # 数据拼接
    result = dict()
    result["code"] = 2
    result["total"] = total
    result["cate_id"] = _cate
    result["cate_name"] = str(cate.objects.get(cate_id=_cate))
    result["news"] = list()
    for one in news:
        result["news"].append({
            "new_id":one.new_id,
            "new_title":str(one.new_title),
            "new_time": one.new_time,
            "new_cate": one.new_cate.cate_name,
            "new_hot_value": news_hot_value[one.new_id] if _cate == "2" or _cate == "1" else 0 ,
            "new_content": str(one.new_content[:100])
        })
    return JsonResponse(result)
```


#### MySQL中的字段

新闻推荐系统中主要的几张数据表如下：

```sql
create table captcha_captchastore (
    id         int auto_increment primary key,
    challenge  varchar(32) not null,
    response   varchar(32) not null,
    hashkey    varchar(40) not null,
    expiration datetime(6) not null,
    constraint hashkey unique (hashkey)
);

create table cate (
    id int auto_increment primary key,
    cate_id   varchar(64) not null,
    cate_name varchar(64) not null,
    constraint cate_id unique (cate_id)
);

create table new (
    id          int auto_increment primary key,
    new_id      varchar(64)  not null,
    new_cate_id int          not null,
    new_time    datetime(6)  not null,
    new_seenum  int          not null,
    new_disnum  int          not null,
    new_title   varchar(100) not null,
    new_content longtext     not null,
    constraint new_id unique (new_id),
    constraint new_new_cate_id_3e5fac50_fk_cate_id
        foreign key (new_cate_id) references cate (id)
)

create table newbrowse (
    id              int auto_increment primary key,
    user_name       varchar(64) not null,
    new_id          varchar(64) not null,
    new_browse_time datetime(6) not null
)

create table newhot (
    id          int auto_increment primary key,
    new_id      varchar(64) not null,
    new_hot     double      not null,
    new_cate_id int         not null,
    constraint new_id unique (new_id),
    constraint newhot_new_cate_id_460738ca_fk_cate_id
        foreign key (new_cate_id) references cate (id)
);

create table newsim (
    id              int auto_increment primary key,
    new_id_base     varchar(64) not null,
    new_id_sim      varchar(64) not null,
    new_correlation double      not null
);

create table newtag (
    id int auto_increment primary key,
    new_tag varchar(64) not null,
    new_id  varchar(64) not null,
    new_hot double      not null
);

create table users_user (
    id int auto_increment primary key,
    name     varchar(128) not null,
    password varchar(256) not null,
    email    varchar(254) not null,
    sex      varchar(32)  not null,
    c_time   datetime(6)  not null,
    constraint email unique (email),
    constraint name unique (name)
);
```