# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from groq import Groq

# app = Flask(__name__)

# # CORS 설정을 더 상세하게 지정
# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://localhost:3000"],
#         "methods": ["GET", "POST", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"],
#         "supports_credentials": True,
#         "expose_headers": ["Content-Range", "X-Content-Range"]
#     }
# })

# # 모든 응답에 CORS 헤더 추가
# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

# # ... (나머지 코드는 동일)

# @app.route('/chat', methods=['POST', 'OPTIONS'])
# def chat():
#     # OPTIONS 요청 처리
#     if request.method == 'OPTIONS':
#         return jsonify({}), 200

#     try:
#         print("Received chat request")
#         data = request.json
#         print(f"Request data: {data}")
        
#         user_message = data.get('message')
#         if not user_message:
#             return jsonify({'error': 'No message provided'}), 400
            
#         response = chatbot.chat(user_message)
#         return jsonify({'response': response})
        
#     except Exception as e:
#         print(f"Error in route handler: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     print("Starting server...")
#     app.run(debug=True, host='0.0.0.0', port=5000)