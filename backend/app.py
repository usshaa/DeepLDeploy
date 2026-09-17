from flask import Flask, request, jsonify
from predict import FaceRecognitionPredictor

app = Flask(__name__) # wsgi - 127.0.0.1:5000/predictor
pre = FaceRecognitionPredictor()

@app.route('/',methods=['GET'])
def index():
    return jsonify({"status":"healthy"}), 200

@app.route('/predictor',methods=['GET'])
def predictor():
    result = pre.predict("images/Alia Bhatt_10.jpg")
    return result

@app.route('/postpred',methods=['POST'])
def postpred():
    if "file" not in request.files:
        return jsonify({"error":"Missing file"}), 400
    
    f = request.files['file'] # txt
    if f.filename == "":
        return jsonify({"error": "No file selceted"}), 400
    try:
        i_bytes = f.read()
        result = pre.predict1(i_bytes)
        return jsonify({
            "success":True,
            "data":result
        }), 200
    except Exception as e:
        return jsonify({
           "success": False,
           "error":f"Inference processing failure: {str(e)}" 
        }), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug=True)