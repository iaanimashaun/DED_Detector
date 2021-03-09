from flask import Flask, redirect, url_for, render_template, request
import os 

app = Flask(__name__, static_url_path='/static')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    return render_template('index.html', content="Testing")



@app.route('/model/')
def model():
    return render_template('model.html')
    

@app.route('/contact/')
def contact():
    return render_template('contact.html')


@app.route('/demo/')
def demo():
    return render_template('demo.html')




@app.route('/upload/', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        destination = '/'.join([target, filename])
        print(destination)
        file.save(destination)

    #return render_template('classify.html')
    return render_template('index.html')




if __name__ == '__main__':
    app.run(port=4555, debug=True)
