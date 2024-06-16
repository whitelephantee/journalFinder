from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# 获取当前文件的目录
base_dir = os.path.abspath(os.path.dirname(__file__))
file_path = os.path.join(base_dir, 'journalFinder_data_final.csv')

# 读取CSV文件
journal_data = pd.read_csv(file_path)

# 关键词搜索功能
def search_by_keyword(keyword, journal_data):
    matching_journals = journal_data[
        journal_data['期刊简介'].str.contains(keyword, na=False, case=False) |
        journal_data['发文领域关键词'].str.contains(keyword, na=False, case=False) |
        journal_data['Journal Name'].str.contains(keyword, na=False, case=False)
    ]

    if matching_journals.empty:
        return pd.DataFrame(), "没有查询到匹配结果，您可以尝试更换关键词或使用摘要进行匹配。"
    
    matching_journals = matching_journals.sort_values(by='jif', ascending=False)
    return matching_journals, None

# 摘要搜索功能
def search_by_abstract(abstract, journal_data):
    # 过滤掉空的 'Aim and Scope'
    journal_data_filtered = journal_data[journal_data['Aim and Scope'].notna()]
    
    if journal_data_filtered.empty:
        return pd.DataFrame(), "没有可用于匹配的期刊数据。"
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([abstract] + journal_data_filtered['Aim and Scope'].tolist())
    
    # 检查 tfidf_matrix 的维度
    if tfidf_matrix.shape[0] < 2:
        return pd.DataFrame(), "没有找到匹配的期刊。"
    
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    journal_data_filtered['similarity'] = cosine_similarities
    matching_journals = journal_data_filtered.sort_values(by='similarity', ascending=False)
    
    return matching_journals, None

@app.route('/')
def index():
    return render_template('index.html', results=pd.DataFrame())

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    search_type = request.form['type']
    
    if search_type == 'keyword':
        results, error_message = search_by_keyword(query, journal_data)
    else:
        results, error_message = search_by_abstract(query, journal_data)
    
    return render_template('index.html', query=query, results=results, search_type=search_type, error_message=error_message)

@app.route('/journal_detail.html')
def journal_detail():
    name = request.args.get('name')
    matching_journal = journal_data[journal_data['Journal Name'].str.lower() == name.lower()]  # 使用小写比较以避免大小写问题
    
    if matching_journal.empty:
        print(f"Journal not found: {name}")  # 打印调试信息
        return render_template('error.html', message="未找到匹配的期刊。")
    
    journal = matching_journal.iloc[0]
    return render_template('journal_detail.html', journal=journal)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
