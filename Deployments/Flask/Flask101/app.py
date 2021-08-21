#import flask
from flask import Flask

# Create application instance - WSGI protocol
app = Flask(__name__)

@app.route('/') # Slash means domain name only
def index():
    return('Hello World')

@app.route('/<name>')
def print_name(name):
    return 'Hello Hello!, {}'.format(name)

if __name__ == '__main__':
    app.run(debug=True)