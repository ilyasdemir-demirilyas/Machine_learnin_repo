data['target_cleaned'] = data['target'].str.split(', ').str.join(',')

data['target_cleaned'].str.get_dummies(sep=',')