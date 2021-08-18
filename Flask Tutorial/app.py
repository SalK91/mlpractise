from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return('Hello World')

@app.route('/<name>')
def print_name(name):
    return 'Hello Hello!, {}'.format(name)

if __name__ == '__main__':
    app.run(debug=True)