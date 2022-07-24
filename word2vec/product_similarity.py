import configparser

import pandas as pd
from gensim.models import Word2Vec



def _preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with raw online data.

    Returns
    -------
    pd.DataFrame that can be converted to list of lists for word3vec training

    """
    
    # Restrict to viewed items only
    df = df.loc[df['event_type']=='view']
    
    # Remove items with NA category code
    df = df.loc[~pd.isna(df['category_code'])]
    
    # Extract viewing date
    df['event_date'] = list(map(lambda x: str(x).split(" ")[0], df['event_time']))
    
    # Remove duplicates in the selected columns
    df = df[['event_date',
             'product_id',
             'category_id',
             'category_code',
             'brand',
             'price',
             'user_id']].drop_duplicates().reset_index()
    
    # Create a subset of data for user-product records where there is at least two viewings
    df_user_brw = df.groupby(['user_id','event_date']).agg({'product_id':'count'}).reset_index()
    df_user_brw = df_user_brw.loc[df_user_brw['product_id']>1]
    
    # Reduce the data records of viewers who have viewed at least two items on a day
    df = pd.merge(left=df, right=df_user_brw, on=['user_id', 'event_date'],
              how='inner', suffixes=('', '_count'))
    
    # Cast product id as str - word2vec works with strs
    df['product_id'] = df['product_id'].astype(str)
    
    return df
    
    

def _train_and_save_model(config_file:str):
    """
    method to train a word2vec model on viewers browsing data
    calls _preprocess_data, saves the model and the preprocessed dataframe
    
    Parameters
    ---------
    config_file: str
        path to the config file for the model and data parameters

    Returns
    -------
    None.

    """
    config = configparser.ConfigParser()
    config.read(config_file)
    
    df = pd.read_csv(config['DATA']['raw_data_file'])
    
    df = _preprocess_data(df)
    
    # Turn viewed products into list of lists
    data = list(df.groupby(['user_id', 'event_date'])['product_id'].apply(list).reset_index()['product_id'])
    
    model = Word2Vec(min_count=int(config['MODEL']['min_count']),
                     vector_size=int(config['MODEL']['vector_size']))
    
    model.build_vocab(data)
    
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Save model and dataframe
    model.save(config['MODEL']['save_as'])
    df[['event_date',
        'product_id',
        'category_id',
        'category_code',
        'brand',
        'price',
        'user_id']].to_csv(config['DATA']['save_as'], index=False)
    
    

def _load_model(config_file:str)->Word2Vec:
    """
    method to load  a pre-trained word2vec model
    
    Parameters
    ---------
    config_file: str
        path to the config file for the model and data parameters
    ------
    Returns the model object loaded from the path as per config.ini
    """
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    return Word2Vec.load(config['MODEL']['save_as'])

def _load_data(config_file:str)->pd.DataFrame:
    """
    method to load  a pre-processed data used to train the model
    
    Parameters
    ---------
    config_file: str
        path to the config file for the model and data parameters
    ------
    ------
    Returns pandas dataframe loaded from the path as per config.ini
    """
    
    config = configparser.ConfigParser()
    config.read(config_file)
    
    return pd.read_csv(config['DATA']['save_as'])
    


def get_similar_product(model:Word2Vec, df:pd.DataFrame, product_id:int)->pd.DataFrame:
    """
    Parameters
    ----------
    model : instance of Word2Vec
    df : dataframe with preprocessed data
    product_id : int
        unique product id for which we need to find similar product ids

    Returns
    -------
    dataframe with similar products

    """
    
    #model = _load_model()
    
    try:
        sim_product = model.wv.most_similar(positive=[str(product_id)])
    
        return df.loc[df['product_id'].isin([int(word[0]) 
                                             for word in sim_product])][[
                                                     'category_code',
                                                     'brand',
                                                     'product_id']].drop_duplicates()
    except KeyError:
        return f"Cannot find the specified product with id {product_id}"
    

def recommend_next_purchase(model: Word2Vec, df:pd.DataFrame, user_id:int)->pd.DataFrame:
    """
    Parameters
    ----------
    model : instance of Word2Vec
    df : dataframe with preprocessed data
    user_id : int
        unique user id for whom we make recommendations
        
    Returns
    -------
    dataframe with recommended products

    """
    
    
    try:
        # Find the products the user browsed
        viewed_products = df.loc[df['user_id']==user_id]['product_id'].unique()
        
        # Get recommendations for next purchase
        output_words = model.predict_output_word([str(product) for product in viewed_products])
        
        return df.loc[df['product_id'].isin([int(word[0]) 
                                             for word in output_words])][[
                                                     'category_code',
                                                     'brand',
                                                     'product_id']].drop_duplicates()
            
    except KeyError:
        return f"Cannot find the specified user with id {user_id}"
        