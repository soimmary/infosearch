# HW4.2
### Инструкция по запуску:
- чтобы посчитать метрику для ВМ25, открыть и запустить файл [bm25.py.py](https://github.com/soimmary/infosearch/blob/main/hw4_2/bm25.py)
- чтобы посчитать метрику для ВМ25, открыть и запустить файл [sbert.py](https://github.com/soimmary/infosearch/blob/main/hw4_2/sbert.py)

### Как решение работает верхнеуровнево:
1. Собираю корпус;
2. Индексирую корпус (в качестве векторизации документов корпуса - слагаемые BM25 или BERT), на выходе получаю посчитанную матрицу Document-Term: документы корпуса для индексации - ответы из датасета.
4. Индексирую запросы, считаю их векторы: запросы - вопросы из датасета;
5. Вычисляю метрику близости пар (BM25 или косинусная близость) от каждого запроса до каждого документа корпуса (с помощью матричных методов);
6. Реализую функцию, которая считает итоговую метрику на топ-5 результатов;
7. В результате вывожу метрику (accuracy).
