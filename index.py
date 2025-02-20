import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

sia=SentimentIntensityAnalyzer()

positive_review_count=int()
negative_review_count=int()
neutral_review_count=int()

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

tweets = [
    "The new policy is great! It will help many people. #government",
    "I hate this new law. It’s terrible and unfair! 😡",
    "The policy changes are okay, but I don’t see much impact.",
    "Absolutely love the new tax reforms! Great work!",
    "Not sure how I feel about this policy, mixed emotions.",
    "This is the worst decision ever made by the government!",
    "I'm neutral about this, doesn't affect me much.",
    "A step in the right direction! More needs to be done though."
]

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

for tweet in tweets:

    # Tokenization and lowercase
    tokens=word_tokenize(tweet.lower())

    # Remove punctuation
    tokens=[word for word in tokens if word.isalnum()]


    tokens=[lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tweet=" ".join(tokens)

    #Sentiment Analysis
    sentiment_score=sia.polarity_scores(tweet)['compound']
    

    if sentiment_score >= 0.5:
        positive_review_count += 1
    elif sentiment_score <= 0.05:
        negative_review_count += 1
    else:
        neutral_review_count += 1



print(f"Postive reviews: {positive_review_count} \nNegative reviews: {negative_review_count} \nNeutral reviews: {neutral_review_count}")









