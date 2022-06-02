import json
from flask import Flask, jsonify,request,make_response
people_count = 2
from flask import Flask

app=Flask(__name__)


@app.route('/peoplenum',methods=['GET','POST'])
def peoplenum():
    method = request.method
    res = make_response(jsonify(people_count=people_count,method=method))  # 设置响应体
    res.status = '200'  # 设置状态码
    res.status = '200'  # 设置状态码
    res.headers['Access-Control-Allow-Origin'] = "*"  # 设置允许跨域
    res.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
    return res

app.run(host='0.0.0.0', port=5000, debug = True)