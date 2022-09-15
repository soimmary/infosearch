# infosearch
HSE course

# HW_1
**Инструкция по запуску:**
- открыть и запустить файл [main.py](https://github.com/soimmary/infosearch/blob/main/hw1/main.py)
**Как решение работает верхнеуровнево:**
1. Собираю все директории txt файлов;
2. Делаю препроцессинг текстовых файлов: удаляю стоп-слова, пунктуацию и лемматизирую – и сохраняю в новую папку _preprocessed-data_;
3. Индексирую данные: 
    - сначала создаю словарь, в котором ключи – названия текстового файла, а значения – список из лемм, взятых из текста;
    - для формата словаря "переворачиваю" словарь так, что ключами становятся лемма, а значением список из названия текстового файла и частотности появления слова в этом файле;
    - для формата матрицы использую CountVectorizer;
4. В модулях [search_in_dict.py](https://github.com/soimmary/infosearch/blob/main/hw1/search_in_dict.py)и [search_in_matrix.py](https://github.com/soimmary/infosearch/blob/main/hw1/search_in_matrix.py) отвечаю на поставленные в домашке вопросы: ищу самые популярные/редкие слова и т.п.

**P.S.**
У меня получилось так, что слова, которые существовали бы в каждом текстовом файле, отсутствуют. Я это могу связать с тем, что на этапе препроцессинга были удалены все стоп-слова, который как раз с высокой вероятностью встречались бы в каждом тексте.
