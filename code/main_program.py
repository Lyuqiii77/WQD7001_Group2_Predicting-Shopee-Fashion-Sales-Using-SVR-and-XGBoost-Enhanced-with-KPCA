# import the libarries and files
import pandas as pd
from imputation_kinda_looks_weird import rating_count, fees_extraction
import datetime
dataset_sample = pd.read_csv("shopee_sample_data.csv")
# define function to check if a number in a str can be converted to a
# float(either character or number)


def isfloat(value):
    try:
        float(value)  # Try to convert the value to float
        return True   # If successful, it's a float
    except ValueError:  # If it fails, it's not a float
        return False


# item_category_details have NA values which are considered
# as float, if float(NA) is true, get the index
dataset_sample.iloc[dataset_sample[
    dataset_sample['item_category_detail'].apply(
                                                lambda x: isfloat(x)
                                                )
                    ].index]
# drope datasets with lots of NA
# (some have the entire row as not availbale)
dataset_sample.dropna(thresh=10, inplace=True, axis=0)
# transform the strings of
# item to lower case to avoid redundancy
dataset_sample['item_category_detail'] = dataset_sample[
    'item_category_detail'
    ].apply(lambda x: x.lower())
# create a list using the separator | within the
# item category detail as an indicator
dataset_sample['item_category_detail'] = dataset_sample[
    'item_category_detail'
    ].apply(lambda x: x.split('|'))
# remove the useless space at the beginning and end
# of the item category to avoid redundancy
dataset_sample['item_category_detail'] = dataset_sample[
    'item_category_detail'
    ].apply(lambda x: [st.strip() for st in x])
# the new item category feature in the dataset is
# a list whose first element
# is the app(shopee in our case)
# and the second and third element of that list represent the type of product
dataset_sample['Type of product'] = dataset_sample[
    "item_category_detail"
    ].apply(lambda x: x[1])
# some product have two types while others have one,
# can be useful for analytics
dataset_sample['Type of product_2'] = dataset_sample[
    "item_category_detail"
    ].apply(lambda x: x[2] if len(x) >= 3 else x[1])
# analysing the type of products available
# in the dataset and chosing the
# ones that are considered fashion
fashion_names = ["women's clothing", "men's clothing", 'muslim fashion',
                 'baby & toys', 'watches', "men's bags & wallets",
                 "women's bags", 'sports & outdoor', 'fashion accessories',
                 "men's shoes", "women's shoes", 'women clothes']
# type of product 2 comes in
# handy when we have vague type
# of product like baby and toys so we pick
# the ones of inetrest
fashion_names_2 = ['baby clothing', 'boys fashion',
                   'girls fashion', 'kids fashion accessories & bags']
# create a new dataset with the elements mentioned above
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
# drop the elements that are
# part of the data set but are not considered fashion
new_set = new_set[new_set['Type of product_2'].apply(
    lambda x: x not in list_to_drop
    )]
# drop useless features such as seller details and seller name(ethic stuff)
columns_to_drop = ['link_ori', 'seller_name', 'sitename',
                   'idElastic', 'idHash',
                   'seller_details', 'id',
                   'pict_link', 'location']
# wrangle and make the dataset more neat for easier handling
new_set.reset_index(inplace=True)
new_set.drop('index', axis=1, inplace=True)
new_set.drop(columns_to_drop, axis=1, inplace=True)
# modify the feature for consistency and format,
# by converting the amount from str to float
new_set.loc[
    :, 'total_sold'] = new_set.loc[
        :, 'total_sold'].apply(
            lambda x: float(x.split('k')[0])*1000 if 'k' in str(x) else float(
                x)
            )
# modify the feature for consistency and format,
# by converting the rating from str to float
new_set.loc[
    :, 'total_rating'] = new_set.loc[
        :, 'total_rating'].apply(
            lambda x: float(x.split('k')[0])*1000 if 'k' in str(x) else float(
                x)
            )
# consistency and format, by converting the amount from str to float
new_set.loc[
    :, 'favorite'] = new_set.loc[
        :, 'favorite'].apply(
            lambda x: float(x.split('k')[0])*1000 if (
                'k' in str(x) and x != 'label_favorite'
                ) else (float(x) if isfloat(x) else x)
            )
# make it so that the bought product were considered part of someone's favorite
new_set.loc[
    new_set.loc[:, 'favorite'].apply(lambda x: not isfloat(x)), 'favorite'
    ] = new_set.loc[
        new_set.loc[:, 'favorite'].apply(
            lambda x: not isfloat(x)), 'total_sold'
        ]
# convert the datetime ti actual datetime format
new_set['date_date'] = new_set["w_date"].apply(
    lambda x: datetime.datetime.strptime((
        str(int(x))[0:4]+'-'+str(int(x))[4:6]+'-'+str(int(x))[6:]
        ), '%Y-%m-%d'
    ).date()
                                             )
# set the actual price that are missing equal to original price
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
# drop useless columns after using them to get valid fearures


def imput(rating,
          df=new_set[['total_sold', 'total_rating', 'favorite',
                      'price_actual', 'price_ori', 'item_rating']]
          ):
    da = df.groupby("item_rating").agg('mean')
    return da.loc[rating,]


new_set.loc[
    new_set['price_actual'].isna(), 'price_actual'
    ] = new_set.loc[
        new_set['price_actual'].isna(), 'item_rating'
        ].apply(lambda x: imput(rating=x))
new_set.loc[
    new_set['price_ori'].isna(), 'price_ori'
    ] = new_set.loc[
        new_set['price_ori'].isna(), 'item_rating'
        ].apply(lambda x: imput(rating=x))

columns_to_drop = ['w_date', 'timestamp', 'desc']
new_set.drop(columns_to_drop, axis=1, inplace=True)
new_set.dropna(axis=0, inplace=True)
# extract valid data from the delivery
# feature : namely where it is from ,
# where it is going and how much was it
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
# drop useless columns
new_set.drop(
    ['delivery', 'item_category_detail', 'specification', 'title'],
    axis=1, inplace=True)
new_set.to_csv('fashion_data_set.csv', index=False)
