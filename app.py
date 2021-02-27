# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 01:08:50 2021

@author: hetul
"""
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
pd.set_option('mode.chained_assignment', None)
contentdf=pd.read_csv('booksdataset.csv')
cdf=pd.read_csv('cdf.csv')
st.title("BOOK RECOMMENDATION APP")
st.sidebar.title("Find Your Books!")
st.subheader('The best is when you find a book you cannot put down')
st.info('This app will help you find your favourite book that takes you to paradise.')
from PIL import Image 
img = Image.open("pic.jpg")  
st.image(img,use_column_width=True)
menu = st.sidebar.selectbox("ARE YOU:",("A NEW USER","ALREADY A USER"))
if menu == "A NEW USER":
    n=st.sidebar.radio("DO YOU KNOW ANY PREVIOUSLY LIKED BOOKS?",("YES","NO"))
    if n == "YES":
        try:
            book=st.sidebar.text_input("ENTER A BOOK YOU LIKED:")
            if st.sidebar.button("Enter",key='enter'):
                st.write("""### HERE ARE A FEW RECOMMENDATIONS FOR YOU:""")
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import linear_kernel
                tfidf = TfidfVectorizer(sublinear_tf=True, min_df=15,ngram_range=(1,2), stop_words='english')
                tfidf_description = tfidf.fit_transform((contentdf["cleaned_description"])) 
                from sklearn.metrics.pairwise import cosine_similarity 
                cos_sim = linear_kernel(tfidf_description, tfidf_description) 
                indices = pd.Series(contentdf.index) 
                def recommendations(title, cosine_sim = cos_sim): 
                    recommended_book = [] 
                    index = indices[indices == title].index[0]
                    similarity_scores = pd.Series(cosine_sim[index]).sort_values(ascending = False) 
                    top_10_books = list(similarity_scores.iloc[1:11].index)
                    for i in top_10_books: 
                        recommended_book.append(list(contentdf.index)[i]) 
                    top_5_books=[]
                    for i in recommended_book:
                        if i not in top_5_books and i!=title:
                            top_5_books.append(i)
                    recommendbook=[]
                                
                    for i in top_5_books:
                        recommendbook.append(contentdf['title'][i])
                    return recommendbook
                a=contentdf.loc[contentdf['title']==book]
                id=a.index[0]
                r=recommendations(id)
                for i,j in zip(r,range(1,len(r)+1)):
                    st.write(j,".",i)
                
            
            #CODE
            
        except IndexError:
                st.write("SORRY! THE WINE IS NOT PRESENT IN OUR DATA")
    else:
        st.write("""### TRY THE MOST POPULAR BOOKS:""")
        st.write("1. Harry Potter and the Order of the Phoenix (Book 5)")
        st.write("2. To Kill a Mockingbird")
        st.write("3. Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))")
        st.write("4. The Secret Life of Bees")
        st.write("5. The Da Vinci Code")
        st.write("6. The Lovely Bones: A Novel)")
        st.write("7. The Red Tent (Bestselling Backlist)")
        st.write("8. The Poisonwood Bible: A Novel")
        st.write("9. Where the Heart Is (Oprah's Book Club (Paperback))")
        st.write("10. Angels &amp; Demons")








        
        #CODE
       
else:
    name = st.sidebar.text_input("ENTER YOUR NAME:")
    user_id =st.sidebar.slider("SELECT YOUR USER ID:", 0, 117) 
    st.sidebar.success('USER ID SELECTED: {}'.format(user_id)) 
     
    if user_id in range(0,118):
        from surprise import Reader
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(cdf[['User_ID','title_id','Rating']], reader)
        svd = SVD()
        cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        def user_rec(id):
            user= cdf[['ISBN','Title','Author','Year_Of_Publication','Publisher','title_id']].copy()
            user = user.reset_index()
# getting full dataset
            data = Dataset.load_from_df(cdf[['User_ID','title_id','Rating']], reader)
            trainset = data.build_full_trainset()
            svd.fit(trainset)
            user['Estimate_Score'] = user['title_id'].apply(lambda x: svd.predict(id, x).est)
            user = user.drop(['index','title_id'], axis = 1)
            user= user.sort_values('Estimate_Score' , ascending = False)
            counts1 = user['Estimate_Score'].value_counts()
            user = user[user['Estimate_Score'].isin(counts1[counts1 == 1].index)]
            return user.head(10)
        Uid=user_id
        details=cdf.loc[cdf['User_ID']==Uid]
        id=details['User_ID'].iloc[0]
        a=user_rec(id)
        a.reset_index(inplace=True)
        a.drop(['index'],axis=1,inplace=True)
        details.reset_index(inplace=True)
        details.drop(['index','User_ID','title_id','ISBN'],axis=1,inplace=True)
        
        title_1=list(details['Title'])
        rat1=list(details['Rating'])
        st.write("""### YOU HAVE RATED THESE BOOKS: """)
        for i,j in zip(title_1,rat1):
            st.write(i,'=>',j)
        
        st.write("""### TOP PICKS FOR YOU:""")
        st.write(a)
    else:
        st.error("""### Invalid Input : Out of Range""")