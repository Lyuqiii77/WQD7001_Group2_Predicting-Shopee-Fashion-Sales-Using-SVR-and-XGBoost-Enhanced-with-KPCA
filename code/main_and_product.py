import pandas as pd
from imputation_kinda_looks_weird import rating_count, fees_extraction
import datetime
import before_cleaning_data_EDA as bfeda
import after_cleaning_EDA as afeda
dataset_sample = pd.read_csv("shopee_sample_data.csv")

bfeda.EDA_pro_GUI(dataset_sample)

def isfloat(value):
    try:
        float(value)  # Try to convert the value to float
        return True   # If successful, it's a float
    except ValueError:  # If it fails, it's not a float
        return False


dataset_sample.iloc[dataset_sample[
    dataset_sample['item_category_detail'].apply(
                                                lambda x: isfloat(x)
                                                )
                    ].index]
dataset_sample.dropna(thresh=10, inplace=True, axis=0)
dataset_sample['item_category_detail'] = dataset_sample[
    'item_category_detail'
    ].apply(lambda x: x.lower())
dataset_sample['item_category_detail'] = dataset_sample[
    'item_category_detail'
    ].apply(lambda x: x.split('|'))
dataset_sample['item_category_detail'] = dataset_sample[
    'item_category_detail'
    ].apply(lambda x: [st.strip() for st in x])
dataset_sample['Type of product'] = dataset_sample[
    "item_category_detail"
    ].apply(lambda x: x[1])
dataset_sample['Type of product_2'] = dataset_sample[
    "item_category_detail"
    ].apply(lambda x: x[2] if len(x) >= 3 else x[1])
fashion_names = ["women's clothing", "men's clothing", 'muslim fashion',
                 'baby & toys', 'watches', "men's bags & wallets",
                 "women's bags", 'sports & outdoor', 'fashion accessories',
                 "men's shoes", "women's shoes", 'women clothes']
fashion_names_2 = ['baby clothing', 'boys fashion',
                   'girls fashion', 'kids fashion accessories & bags']
new_set = dataset_sample[
    (dataset_sample['Type of product'].isin(fashion_names)) | (
        dataset_sample['Type of product_2'].isin(fashion_names_2)
        )]
list_to_drop = ['baby gear', 'feeding & nursing',
                'bath & toiletries', 'formula & food',
                'toys & education', 'nursery',
                'diapers & potties',
                'baby & toys', 'baby & toddler play',
                'kids sports & outdoor play',
                'outdoor & adventure', 'exercise & fitness equipment',
                'sports & outdoor', 'badminton', 'cycling & skates',
                'kids health & skincare', 'baby safety',
                'acrobatic & martial arts',
                'basketball', 'running',
                'racket sports', 'maternity care',
                'stick & ball games', 'fishing',
                'water sports', 'performance wear', 'football & futsal']
new_set = new_set[new_set['Type of product_2'].apply(
    lambda x: x not in list_to_drop
    )]
columns_to_drop = ['link_ori', 'seller_name', 'sitename',
                   'idElastic', 'idHash',
                   'seller_details', 'id',
                   'pict_link', 'location']
new_set.reset_index(inplace=True)
new_set.drop('index', axis=1, inplace=True)
new_set.drop(columns_to_drop, axis=1, inplace=True)
new_set.loc[
    :, 'total_sold'] = new_set.loc[
        :, 'total_sold'].apply(
            lambda x: float(x.split('k')[0])*1000 if 'k' in str(x) else float(
                x)
            )
new_set.loc[
    :, 'total_rating'] = new_set.loc[
        :, 'total_rating'].apply(
            lambda x: float(x.split('k')[0])*1000 if 'k' in str(x) else float(
                x)
            )
new_set.loc[
    new_set.loc[:, 'favorite'].apply(lambda x: not isfloat(x)), 'favorite'
    ] = new_set.loc[
        new_set.loc[:, 'favorite'].apply(
            lambda x: not isfloat(x)), 'total_sold'
        ]
new_set.loc[
    :, 'favorite'] = new_set.loc[
        :, 'favorite'].apply(
            lambda x: float(x.split('k')[0])*1000 if 'k' in str(x) else float(
                x)
            )

new_set['date_date'] = new_set["w_date"].apply(
    lambda x: datetime.datetime.strptime((
        str(int(x))[0:4]+'-'+str(int(x))[4:6]+'-'+str(int(x))[6:]
        ), '%Y-%m-%d'
    ).date()
                                             )
new_set.loc[
    new_set['item_rating'].isna(), 'item_rating'
    ] = new_set.loc[
        new_set['item_rating'].isna(), 'price_actual'
    ].apply(lambda row: rating_count(row))
new_set.loc[
    new_set['price_ori'].isna(), 'price_ori'
    ] = new_set.loc[
        new_set['price_ori'].isna(), 'price_actual'
        ]
columns_to_drop = ['w_date', 'timestamp', 'desc']
new_set.drop(columns_to_drop, axis=1, inplace=True)
new_set.dropna(axis=0, inplace=True)
new_set['from'] = new_set['delivery'].apply(
    lambda x: ('|'.join(x.lower().split('shipping')
                        ).split('from')[1].split('to')[0]
               ).strip() if 'from' in x else 'local'
    )
new_set['from'] = new_set['from'].apply(
    lambda x: (x.split("no")[0]) if 'options available' in x else (
        'overseas' if 'label' in x else x
        )
    )
new_set["to"] = new_set['delivery'].apply(
    lambda x: (
        ('|'.join(x.lower().split('shipping')
                  ).split('to'))[1].split('|')[0]
        ).strip() if 'to' in x else (
                           'delivery no available'
                           )
                       )
new_set['to'] = new_set['to'].apply(
    lambda x: ('kuala lumpur') if 'kl city' in x else x
    )

new_set['fees'] = new_set['delivery'].apply(
    lambda x: x.split('fee')[1].strip() if 'fee' in x else 'unavailable'
    )
new_set['fees'] = new_set['fees'].apply(lambda x: fees_extraction(x))
new_set.drop(
    ['delivery', 'item_category_detail', 'specification', 'title'],
    axis=1, inplace=True)

afeda.EDA_pro_GUI(new_set)
new_set.to_csv('fashion_data_set.csv', index=False)
