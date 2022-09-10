import yaml,os
#Programmer: Tim Tyree
#Date: 9.9.2022
#forked from article:
# https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started

def save_to_yaml(save_fn,mdict):
    '''saves dict instance to .yaml.

    Example Usage:
documents = save_to_yaml(input_fn,mdict)
    '''
    with open(save_fn, 'w') as file:
        documents = yaml.dump(mdict, file)
        return documents

def read_yaml(input_fn):
    '''returns dict instance.  input_fn is a string locating a yaml file.

    Example Usage:
dictionary = read_yaml(input_fn)
    '''
    stream = open(input_fn, 'r')
    dictionary = yaml.load(stream)
    return dictionary

##################################
# aliases
##################################
def load_from_yaml(input_fn):
    '''returns dict instance.  input_fn is a string locating a yaml file.

    Example Usage:
dictionary = load_from_yaml(input_fn)
    '''
    return read_yaml(input_fn)

if __name__ == '__main__':
    stream = open("foo.yaml", 'r')
    dictionary = yaml.safe_load_all(stream)

    for doc in dictionary:
        print("New document:")
        for key, value in doc.items():
            print(key + " : " + str(value))
            if type(value) is list:
                print(str(len(value)))
