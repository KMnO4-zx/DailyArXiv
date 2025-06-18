import sys
import time
import pytz
from datetime import datetime

# 从utils模块导入所需的功能函数
from utils import get_daily_papers_by_keyword_with_retries, generate_table, back_up_files,\
    restore_files, remove_backups, get_daily_date


# 设置北京时区
beijing_timezone = pytz.timezone('Asia/Shanghai')

# 注意: arXiv API有时会意外返回空列表

# 获取当前北京时间的日期，格式为"2021-08-01"
current_date = datetime.now(beijing_timezone).strftime("%Y-%m-%d")
# 从README.md文件中获取最后更新日期
with open("README.md", "r") as f:
    while True:
        line = f.readline()
        if "Last update:" in line: break  # 找到包含更新日期的行
    last_update_date = line.split(": ")[1].strip()  # 提取日期部分
    # 如果今天已经更新过，则退出程序
    if last_update_date == current_date:
        sys.exit("Already updated today!")

# 定义搜索关键词列表
keywords = ["Geoscience", "Spatial", "Time", "Trajectory", "City"] # TODO 添加更多关键词

max_result = 100  # 每个关键词从arXiv API查询的最大结果数
issues_result = 15  # 在issue中包含的最大论文数量

# 所有可用列: Title, Authors, Abstract, Link, Tags, Comment, Date
# 固定列: ["Title", "Link", "Date"]

# 定义要显示的列名
column_names = ["Title", "Link", "KiMi", "Abstract", "Date", "Comment"]

# 备份README.md和ISSUE_TEMPLATE.md文件
back_up_files()

# 写入README.md文件
f_rm = open("README.md", "w")  # 打开README.md文件用于写入
f_rm.write("# Daily Papers\n")  # 写入标题
# 写入项目说明和最后更新日期
f_rm.write("The project automatically fetches the latest papers from arXiv based on keywords.\n\nThe subheadings in the README file represent the search keywords.\n\nOnly the most recent articles for each keyword are retained, up to a maximum of 100 papers.\n\nYou can click the 'Watch' button to receive daily email notifications.\n\nLast update: {0}\n\n".format(current_date))

# 写入ISSUE_TEMPLATE.md文件
f_is = open(".github/ISSUE_TEMPLATE.md", "w")  # 打开ISSUE_TEMPLATE.md文件用于写入
f_is.write("---\n")  # 写入YAML前端分隔符
# 设置issue标题，包含最新论文数量和日期
f_is.write("title: Latest {0} Papers - {1}\n".format(issues_result, get_daily_date()))
f_is.write("labels: documentation\n")  # 设置issue标签
f_is.write("---\n")  # 写入YAML后端分隔符
# 添加指向GitHub仓库的链接提示
f_is.write("**Please check the [Github](https://github.com/KMnO4-zx/DailyArXiv) page for a better reading experience and more papers.**\n\n")

# 遍历每个关键词，获取相关论文并生成表格
for keyword in keywords:
    # 在README和issue模板中写入关键词作为二级标题
    f_rm.write("## {0}\n".format(keyword))
    f_is.write("## {0}\n".format(keyword))
    
    # 确定搜索逻辑：单个词使用AND逻辑（标题和摘要都必须包含关键词），多个词使用OR逻辑
    if len(keyword.split()) == 1: 
        link = "AND"  # 对于单个词的关键词，搜索同时包含该关键词在标题和摘要中的论文
    else: 
        link = "OR"  # 对于多个词的关键词，使用OR逻辑
    
    # 尝试获取论文数据，带有重试机制
    papers = get_daily_papers_by_keyword_with_retries(keyword, column_names, max_result, link)
    
    # 如果获取论文失败，恢复备份文件并退出程序
    if papers is None:  # 获取论文失败
        print("Failed to get papers!")
        f_rm.close()
        f_is.close()
        restore_files()  # 恢复备份文件
        sys.exit("Failed to get papers!")
    
    # 为README生成完整论文表格
    rm_table = generate_table(papers)
    # 为issue生成精简版表格（不包含摘要），且只包含前issues_result篇论文
    is_table = generate_table(papers[:issues_result], ignore_keys=["Abstract"])
    
    # 将表格写入相应文件
    f_rm.write(rm_table)
    f_rm.write("\n\n")
    f_is.write(is_table)
    f_is.write("\n\n")
    
    # 休眠5秒，避免被arXiv API封锁
    time.sleep(5)

# 关闭文件
f_rm.close()
f_is.close()
# 删除备份文件
remove_backups()
