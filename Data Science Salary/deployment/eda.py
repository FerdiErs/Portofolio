import streamlit as st 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
from PIL import Image
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS

st.set_page_config(
    page_title = 'Data Science Salary Estimator'
)

def run():

    # Membuat Title 
    st.title('Data Science Salary Estimator')

    #Sub header 
    st.subheader('Description for Data Science Salary Estimator') 

    # Insert Gambar 
    image = Image.open('gaji.jpg')
    st.image(image, caption ='SALARY')

    #description 
    st.write('The goals of this salary estimator')
    st.write('as a data scientist i want to know if im getting the decent salary from the company, so i created this machine learning model to predict salary for jobs in data world.')
    st.write('I hope with this salary estimator can help fellas data to see if they also get a decent salary or not')
    st.markdown('---')
    
    st.write('This page is created to show the visualization of the dataset')

    st.markdown('---')

    st.write('Description')
    st.write('Experience Level')
    st.write('EN, which is Entry-level. MI, which is Mid-level. SE, which is Senior-level. EX, which is Executive-level.')

    st.write('Employment Type')
    st.write('FT, which is Full Time. PT, which is Part Time. CT, which is Contract. FL, which is Freelance.')

    st.write('Remote Ratio')
    st.write('100, which is Full remote. 50, which is hybrid. 0, which is on site.')

    st.markdown('---')



    #show dataframe 
    data = pd.read_csv('https://raw.githubusercontent.com/FerdiErs/SQL/main/DataScienceSalaries.csv')
    st.dataframe(data)


    #membuat histogram salary 
    st.write('### Histogram Salary')
    fig = plt.figure(figsize=(10,5))
    sns.histplot(data['salary_in_usd'], kde=True, bins=40)
    plt.title('Histogram of salary in usd')
    st.pyplot(fig)

    #membuat pie chart experience 
    st.write('### Experince Distribution')
    exp = data.experience_level.value_counts()
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct
    fig = plt.figure(figsize=(5,5))
    exp.plot.pie(autopct=make_autopct(exp))
    plt.title('Experince Level Distribution')
    st.pyplot(fig)

    
    #barchart posisi dengan gaji terbesar
    st.write('### 5 Role with highest paycheck')
    work_rate = data.groupby(['job_title'])['salary_in_usd'].mean()
    work = work_rate.nlargest(5)
    fig = plt.figure(figsize=(15,5))
    work.plot(kind = "bar")
    plt.title('5 Role with Highest Paycheck')
    st.pyplot(fig)


    # negara dengan gaji tertinggi 
    st.write('### Country with highest paycheck')
    location_payrate = data.groupby(['company_location'])['salary_in_usd'].sum()
    lar = location_payrate.nlargest(5)
    fig = plt.figure(figsize=(15,5))
    lar.plot(kind = "bar")
    plt.title('5 Countries Highest Paycheck')
    st.pyplot(fig)


    # popular job
    st.write('### TOP 10 JOBS')
    job = data.groupby(['job_title'])['job_title'].count()
    top_job = job.nlargest(10)
    fig = plt.figure(figsize=(12,6))
    plt.xticks(rotation=0)
    plt.title("Top 10 Jobs")
    plt.ylabel('Job Titles')
    plt.xlabel('Counts')
    sns.barplot(y=top_job.index, x= top_job.values)
    st.pyplot(fig)
    

    #wordcloud 
    # see most job with word cloud
    text = " ".join(i for i in data.job_title)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=1600, height=800).generate(text)
    fig = plt.figure( figsize=(15,10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)



if __name__== '__main__':
    run()