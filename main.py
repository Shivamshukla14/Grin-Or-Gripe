from flask import Flask, render_template, url_for, request, Blueprint, redirect, flash ,session
from flask_bootstrap import Bootstrap

#Disabling oneDNN (Optional)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#Import for app1
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import re
import numpy as np


#Import for app2
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

#Import for app3
from textblob import TextBlob, Word
import random
import time

#Import for app4
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, DataRequired, Length, Email
from werkzeug.utils import secure_filename


#Import for app6
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
import smtplib, ssl

#Import for record maintainance
import csv
from datetime import datetime


#Loading the Environment Variables for app4
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#Downloading stopwords
# nltk.download('stopwords')


#basic requirements for app5
app6_analyzer = SentimentIntensityAnalyzer()
app6_API_KEY = "news_api_key"
country = "in"
app6_url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={app6_API_KEY}"


#Initializing the app------------------------------------------------------
app = Flask(__name__)


app.secret_key = 'chatbot'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')


#Initializing blueprint for each app---------------------------------------
app1_bp = Blueprint('app1', __name__)
app2_bp = Blueprint('app2', __name__)
app3_bp = Blueprint('app3', __name__)
app4_bp = Blueprint('app4', __name__)
app5_bp = Blueprint('app5', __name__)


app1_csv_filename = "static/csv/app1_input_data.csv"
app2_csv_filename = "static/csv/app2_input_data.csv"
app3_csv_filename = "static/csv/app3_input_data.csv"
app5_csv_filename = "static/csv/app5_input_data.csv"


# Function to write input data with current date and time to a CSV file
def write_to_csv(sendto, app5_csv_filename):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(app5_csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_datetime, sendto])

def write_to_csv_1(input_text, output_data, csv_filename):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_datetime, input_text, output_data])

def write_to_csv_2(input_text, output_data, csv_filename):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_datetime, input_text, *output_data])


#login page
class LoginForm(FlaskForm):
    email = StringField('Email', validators = [DataRequired(), Email()])
    password = PasswordField('Password', validators = [DataRequired()])
    remember = BooleanField('Remeber me!')
    submit = SubmitField('Login')

#Initiation Flask form for app4
class UploadForm(FlaskForm):
    file = FileField('PDF File', validators=[InputRequired()])
    submit = SubmitField('Submit')

#admin details
users = {'admin@gog.com': 'password'}

#For homepafge rendering----------------------------------------------------
@app.route('/home')
@app.route('/')
def home():
    session.clear()
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@gog.com' and form.password.data == 'password':
            # Store login status in session for access control
            session['logged_in'] = True
            return redirect(url_for('admin_database'))
        else:
            login_msg = "Login Unsuccessful! Invalid Email ID or Password!"
            return render_template('login.html', title="admin login", login_msg=login_msg, form=form)
    return render_template('login.html', title="admin login", form=form)


@app.route('/submit', methods=['GET','POST'])
def submit_form():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        suggestion = request.form['suggestion']
        
        with open('static/csv/contact_us.csv', 'a', newline='') as csvfile:
            fieldnames = ['Full Name', 'Email', 'Suggestion']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if csvfile.tell() == 0:
                writer.writeheader()
            
            writer.writerow({'Full Name': full_name, 'Email': email, 'Suggestion': suggestion})
        
        return render_template('home.html', contact_message='Thankyou for your valuable!')
    else:
        return render_template('home.html')

@app.route('/admin_database')
def admin_database():
    # Check if user is logged in using session data
    if 'logged_in' not in session:
        return redirect(url_for('login')) 

    csv_data = [
        ('static/csv/app1_input_data.csv', 'TwitMeter✨'),
        ('static/csv/app2_input_data.csv', 'TwiMeter++'),
        ('static/csv/app3_input_data.csv', 'Veritas🛒'),
        ('static/csv/app5_input_data.csv', 'Posify💌'),
        ('static/csv/contact_us.csv', 'Customer Feedback😋')
    ]

    return render_template('database.html', csv_data=csv_data, title = "GOG - admin")

@app.route('/logout')
def logout():
    # Clear session data to log out user
    session.clear()
    return redirect(url_for('login'))



#Code for app1 (Sentiment Analysis using Pickle)----------------------------
#Initializing the required variables
loaded_vectorizer = None
loaded_model = None
stemmer = PorterStemmer()
emotion_pipeline = pipeline('sentiment-analysis', model = 'arpanghoshal/EmoRoBERTa')

#Function for text pre-processing
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def get_emotion_labels(text):
    emotion_labels = emotion_pipeline(text)
    emotion = emotion_labels[0]['label']
    score = emotion_labels[0]['score']
    return emotion, score

#Opening the pickle file to load the vectorizer and model
with open("models/model_and_vectorizer.pkl", "rb") as f:
    data = pickle.load(f)
    loaded_vectorizer = data['vectorizer']
    loaded_model = data['model']


#Routing to app 1---------------------------------------------------------------
@app1_bp.route('/app1/')
def app1_home():
    return render_template('app1.html', title = "GOG - TwitMeter✨")


@app1_bp.route('/app1/predict', methods = ['POST', 'GET'])
def app1_predict():
    start = time.time()
    if request.method == 'POST':
        user_input = request.form["tweet"]
        text = user_input
        processed_input = stemming(user_input)
        X_new = np.array([processed_input]).reshape(1, -1)
        processed_text = ' '.join(X_new[0])
        X_new_transformed = loaded_vectorizer.transform([processed_text])
        prediction = loaded_model.predict(X_new_transformed)
        sentiment = 'Negative' if prediction[0] == 0 else "Positive"
        end = time.time()
        final_time = round(end-start, 2)
        write_to_csv_1(user_input, sentiment, app1_csv_filename)
        return render_template("app1.html", text = text, sentiment = sentiment, final_time = final_time, title = "GOG - TwitMeter Result")
    else:
        return redirect(url_for('app1_home'))
    

# app1 end here---------------------------------------------------------------------
    
# app2 starts here------------------------------------------------------------------
#Code for app2 (Sentiment Analysis using pre trained model)

#Initializing the required variable
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
model = None
tokenizer = None


#loading the model and tokenizer
try:
    model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
    tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
except Exception as e:
    print(f"Error loading model/tokenier: {e}")
    app.config['model_error'] = True #Flag for error handling in teamplates

#Defining the step for text preprocessing
def preprocess_tweet(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'  #Replace the mentions with '@user'
        elif word.startswith('http') or word.startswith('www'):
            word = 'http'   #Replace the URLs with 'http'
        tweet_words.append(word)
    tweet_proc = ' '.join(tweet_words)
    return tweet_proc


@app2_bp.route('/app2/')
def app2_analyze_sentiment():
    return render_template('app2.html', title = "GOG - Twitmeter++")

@app2_bp.route('/app2/predict', methods = ['GET', 'POST'])
def app2_analyze_sentiment_result():
    start = time.time()
    if request.method == 'POST':
        tweet = request.form["tweet"]
        text = tweet
        tweet_proc = preprocess_tweet(tweet)
        if not app.config.get('model_error'):
            try:
                encoded_tweet = tokenizer(tweet_proc, return_tensors = 'pt')
                output = model(**encoded_tweet)
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)
                emotion_label, emotion_score = get_emotion_labels(tweet)
                emotion_score = f"{(emotion_score * 100):.2f}"
                labels = ['Negative', 'Neutral', 'Positive']
                sentiment_results = [(label, f"{score:.2f}%") for label, score in zip(labels, scores*100)]
                print(sentiment_results)
                print(emotion_label)
                print(emotion_score)
                end = time.time()
                final_time = round(end-start, 2)
                output_data = [emotion_label,emotion_score]
                for label, score in sentiment_results:
                    output_data.extend([score])
                write_to_csv_2(tweet, output_data, app2_csv_filename)
                return render_template("app2.html",text = text ,sentiment_results = sentiment_results, emotion_label = emotion_label, 
                                       emotion_score = float(emotion_score), final_time = final_time, title = "GOG - TwitMeter++ Result")
            except Exception as e:
                print(f"Error during sentiment analysis: {e}")
                return render_template("app2.html")
        else:
            return render_template("app2.html")
    else:
        return render_template("app2.html")


#App 2 ends here------------------------------------------------------------------------
#App 3 starts here---------------------------------------------------------------------
@app3_bp.route('/app3/')
def app3_home():
    return render_template('app3.html', title = "GOG - Veritas🛒")

@app3_bp.route('/app3/analyse', methods = ['POST'])
def app3_analyse():
    start = time.time()
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        #NLP stuff
        blob = TextBlob(rawtext)
        received_text2 = blob
        blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
        blob_sentiment, blob_subjectivity = round(blob_sentiment, 2), round(blob_subjectivity, 2)
        number_of_tokens = len(list(blob.words))
        #Extracting Main Points
        nouns = list()
        for word, tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())
                len_of_words = len(nouns)
                rand_words = random.sample(nouns, len(nouns))
                final_word = list()
                for item in rand_words:
                    word = Word(item).pluralize()
                    final_word.append(word)
                    summary = final_word
                    end = time.time()
                    final_time = end-start
        output_data = [blob_sentiment,blob_subjectivity]
        for label in summary:
            output_data.append(label)
        print(output_data)
        write_to_csv_2(rawtext, output_data, app3_csv_filename)
    
    return render_template('app3.html', received_text = received_text2, number_of_tokens = number_of_tokens,
                           blob_sentiment = blob_sentiment, blob_subjectivity = blob_subjectivity, summary = summary,
                           final_time = final_time, len_of_words = len_of_words, title = "GOG - Veritas Result")


#app3 ends here------------------------------------------------------------------------
#app4 starts here-----------------------------------------------------------------------

#Definig the functions for text preprocessing of app4

#Creating the functions for text preprocessing

def get_pdf_text(pdf_docs):
    text=""
    # for pdf in pdf_docs:
    #     pdf_reader= PdfReader(pdf)
    for page in pdf_docs.pages:        #Reindent if not in use
        text += page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=3000)
    chunks = text_splitter.split_text(text)
    return chunks



def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.6)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)

    return response


@app4_bp.route('/app4/', methods=["GET", "POST"])
def app4_home():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        session["filename"] = filename
        loader = PdfReader(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        text_data = get_pdf_text(loader)
        chunks = get_text_chunks(text_data)
        get_vector_store(chunks)
    return render_template("app4.html", form = form, filename = session.get("filename", None), title = "GOG - DocuBot📚")
    

@app4_bp.route("/app4/chat", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    # Let's chat for 5 lines
    for step in range(5):
        response = user_input(text)
        print(response['output_text'])
        return response['output_text']



#app4 ends here------------------------------------------------------------------------

#app5 starts here----------------------------------------------------------------------

def sendmail(message, sendto):
        host = "smtp.gmail.com"
        port = 465
        context = ssl.create_default_context()

        username = "your_email"  # Replace with your email
        password = "your_password"  # Replace with your password (store securely)

        try:
            with smtplib.SMTP_SSL(host, port, context=context) as server:
                server.login(username, password)
                server.sendmail(username, sendto, message)
                return "Email sent successfully!"
        except Exception as e:
            return f"Error sending email: {e}"
        

@app5_bp.route('/app5/')
def app5_home():
    return render_template("app5.html", title = "GOG - Posify💌")


@app5_bp.route('/app5/mailsent', methods=['POST'])
def process():
    sendto = request.form['sendto']
    write_to_csv(sendto, app5_csv_filename)
    # Make request
    req = requests.get(app6_url)

    # Store data as dictionary
    content = req.json()

    # News sending to mail
    news = "Subject: Grin or Gripe News\n"

    # Organising the Data
    no_of_news = 0
    for data in content["articles"]:
        # Check if description exists before processing
        if data["description"] is not None:
            positive_value = app6_analyzer.polarity_scores(data['description'])['pos']
            negative_value = app6_analyzer.polarity_scores(data['description'])['neg']

            # Select and add news with sentiment check
            if data["title"] is not None and positive_value >= negative_value and no_of_news != 5:
                news += f"\nTitle: {data['title']}\n"
                news += f"Description: {data['description']}\n"
                news += f"URL: {data['url']}\n\n"
                no_of_news += 1
        else:
            print("Article has no description")  # Handle articles with missing descriptions

    if no_of_news > 0:
      email_result = sendmail(news.encode('utf-8'), sendto)
      return render_template('app5.html', message=email_result, title = "GOG - mail sent!")
    else:
        return render_template('app5.html', message="No positive news articles found to send.", title = "GOG - no news today!")

#app5 ends here------------------------------------------------------------------------


#Register Blueprints with the main app---------------------------------------------------
app.register_blueprint(app1_bp)
app.register_blueprint(app2_bp)
app.register_blueprint(app3_bp)
app.register_blueprint(app4_bp)
app.register_blueprint(app5_bp)

if __name__ == '__main__':
    app.run(debug = True)