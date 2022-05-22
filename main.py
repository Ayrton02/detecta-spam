import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from spam import Spam


def main():
    read_email_file()


def read_email_file():
    path = input(
        "Digite o caminho do arquivo de texto para ser classificado:\n")
    try:
        file = open(path, "r")
    except FileNotFoundError:
        print("Não foi possível abrir o arquivo no caminho indicado")
        return

    content = file.read()
    file.close()
    data = organize_data(content)
    make_prediction(data)


def make_prediction(content):
    data = pd.read_csv('spambase.csv')
    base = data.to_numpy()
    row, col = data.shape
    Y = base[:, -1]
    X = base[:, 0:col-1]

    nb = MultinomialNB()
    nb.fit(X, Y)

    prediction = nb.predict([content])
    accuracy = nb.predict_proba([content])

    print("O email foi classificado como",
          "NÃO SPAM" if prediction[0] < 1 else "SPAM")
    print(
        f'A precisão NÃO SPAM é{accuracy[0][0]: .2%} e de SPAM é{accuracy[0][1]: .2%}')


def organize_data(email_content):
    text_length = len(re.findall(
        r'[a-zA-Z]+|[0-9]+|[!#\$\(\[;]+', email_content))

    sp = Spam(
        calc_string_frequency('make',  email_content,  text_length),
        calc_string_frequency('address',  email_content,  text_length),
        calc_string_frequency('all',  email_content,  text_length),
        calc_string_frequency('3d',  email_content,  text_length),
        calc_string_frequency('our',  email_content,  text_length),
        calc_string_frequency('over',  email_content,  text_length),
        calc_string_frequency('remove',  email_content,  text_length),
        calc_string_frequency('internet',  email_content,  text_length),
        calc_string_frequency('order',  email_content,  text_length),
        calc_string_frequency('mail',  email_content,  text_length),
        calc_string_frequency('receive',  email_content,  text_length),
        calc_string_frequency('will',  email_content,  text_length),
        calc_string_frequency('people',  email_content,  text_length),
        calc_string_frequency('report',  email_content,  text_length),
        calc_string_frequency('addresses',  email_content,  text_length),
        calc_string_frequency('free',  email_content,  text_length),
        calc_string_frequency('business',  email_content,  text_length),
        calc_string_frequency('email',  email_content,  text_length),
        calc_string_frequency('you',  email_content,  text_length),
        calc_string_frequency('credit',  email_content,  text_length),
        calc_string_frequency('your',  email_content,  text_length),
        calc_string_frequency('font',  email_content,  text_length),
        calc_string_frequency('000',  email_content, text_length),
        calc_string_frequency('money',  email_content,  text_length),
        calc_string_frequency('hp',  email_content,  text_length),
        calc_string_frequency('hpl',  email_content,  text_length),
        calc_string_frequency('george',  email_content,  text_length),
        calc_string_frequency('650',  email_content, text_length),
        calc_string_frequency('lab',  email_content,  text_length),
        calc_string_frequency('labs',  email_content,  text_length),
        calc_string_frequency('telnet',  email_content,  text_length),
        calc_string_frequency('857',  email_content, text_length),
        calc_string_frequency('data',  email_content,  text_length),
        calc_string_frequency('415',  email_content, text_length),
        calc_string_frequency('85',  email_content, text_length),
        calc_string_frequency('technology',  email_content,  text_length),
        calc_string_frequency('1999',  email_content, text_length),
        calc_string_frequency('parts',  email_content,  text_length),
        calc_string_frequency('pm',  email_content,  text_length),
        calc_string_frequency('direct',  email_content,  text_length),
        calc_string_frequency('cs',  email_content,  text_length),
        calc_string_frequency('meeting',  email_content,  text_length),
        calc_string_frequency('original',  email_content,  text_length),
        calc_string_frequency('project',  email_content,  text_length),
        calc_string_frequency('re',  email_content,  text_length),
        calc_string_frequency('edu',  email_content,  text_length),
        calc_string_frequency('table',  email_content,  text_length),
        calc_string_frequency('conference',  email_content,  text_length),
        calc_special_frequency(';',  email_content),
        calc_special_frequency('\(',  email_content),
        calc_special_frequency('\[',  email_content),
        calc_special_frequency('!',  email_content),
        calc_special_frequency('\$',  email_content),
        calc_special_frequency('#',  email_content),
        calc_capitals_average(email_content),
        capitals_lenght_longest(email_content),
        capitals_lenght_total(email_content)
    )
    return sp.toList()


def calc_string_frequency(word, content, length):
    result = len(re.findall(word.upper(), content.upper()))

    return result/length


def calc_special_frequency(special, content):
    length = len(re.findall(
        r'[a-zA-Z]+|[0-9]+|[!#\$\(\[;]', content))
    special_count = len(re.findall(special, content))

    try:
        return special_count/length
    except:
        return 0.0


def calc_capitals_average(content):
    capitals = re.findall(r'\b[A-Z]{2,}\b', content)
    capitals_count = len(capitals)
    total_chars = 0
    for c in capitals:
        total_chars += len(c)

    try:
        return total_chars/capitals_count
    except:
        return 0.0


def capitals_lenght_longest(content):
    capitals = re.findall(r'\b[A-Z]{2,}\b', content)
    longest = 0
    for c in capitals:
        if (longest < len(c)):
            longest = len(c)

    return longest


def capitals_lenght_total(content):
    capitals = re.findall(r'[A-Z]+', content)
    total_capitals = 0
    for cap in capitals:
        for c in cap:
            total_capitals += len(re.findall(r'[A-Z]', c))

    return total_capitals


if __name__ == '__main__':
    main()
