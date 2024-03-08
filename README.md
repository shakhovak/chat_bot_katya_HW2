# HW2 generative model based chat-bot

**Задание**: необходимо разработать чат-бот, используя генеративный подход. Бот должен вести диалог как определенный персонаж сериала, имитируя стиль и манеру конкретного персонажа сериала.

## Данные
В качестве основы для чат-бота я взяла скрипты к сериалу "Теория большого взрыва", которые есть на Kaggle и можно загрузить по [ссылке](https://www.kaggle.com/code/lydia70/big-bang-theory-tv-show/input).

Основной персонаж - Шелдон Купер :)

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/b4e0e914-3070-483f-a88a-30fae33a61f7)

Данные на kaggle уже удобно разделены на реплики каждого персонажа и на отдельные сцены (ниже принт-скрин данных)

![image](https://github.com/shakhovak/chat_bot_katya/assets/89096305/100d2802-4837-40d9-95ad-c41034e184fb)

### Данные для retrieval-элемента чат-бота

Первоначальная обработка данных схожа с той, что я делала для предыдущего домашнего задания при подготовке retrieval-based чат-бота. 

Обработка данных подразумевает следующие шаги:
- отбор реплик персонажа в качестве ответов / answers. Именно из этих реплик будет выбирать бот свой ответ на высказывание пользователя.
- выделение предшествующей фразы как вопроса. Если это фраза первая в сцене, то это поле будет пустым.
- отбор предыдущих реплик как контекста диалога (ограничение не более 5 фраз в контексте). Если фраза первая в сцене, то контекст также будет пустым. Контекст - идущие подряд предложения. Я не стала разбивать на диалоги, как предлагалось на семинаре.
- сохранение файла в pickle формат для последующего использования алгоритмом (файл scripts.pkl)

Так как я буду использовать дополнение генерации retrieval данными по косинусной близости, то все начальные данные векторизую в базу данных (файл scripts_vectors.pkl). Для векторизации использую обученную в предыдущем задании модель bi-encoder, которую я сохранила в мой репозиторий на Hugging Face ([ссылка](https://huggingface.co/Shakhovak/chatbot_sentence-transformer)). Детально про обучение этой модели - в предыдущем задании ([ссылка на репозиторий](https://github.com/shakhovak/chat_bot_katya/tree/master))

Функции для обработки данных находятся в файле ```utils.py```:
1. ```scripts_rework_ranking``` - перерабатывает файл, как описано выше и сохраняет их в pkl формате
2. ```encode_df_save``` - использует переработанный файл и векторизует его, а вектора уже сохраняет в базу
   
### Данные для генеративной модели

Для обучение генеравной модели я использую первоначальную переработку как и для retrieval-данных, описанную выше. Дополнительно, я разбиваю весь полученный контекст на части: если в контексте есть три предложения, то я в итоге получаю 4 семпла для данных:

- ответ + вопрос + предложение 3 +  предложение 2 + предложение 1
- ответ + вопрос + предложение 3 +  предложение 2
- ответ + вопрос + предложение 3
- ответ + вопрос

Таким образом, получается порядка 50 тыс. семплов для обучения. Обработанные и дополненные таким образом данные сохранены в файл scripts_reworked.pkl.

Функции для обработки данных находятся в файле ```utils.py```:
1. ```scripts_rework``` - перерабатывает файл, как описано выше и сохраняет их в pkl формате

## Архитектура чат-бота

Схематично процесс работы чат-бота представлен на рисунке ниже.

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/7fa47235-c805-439f-bdea-c4ed203c12d4)

### Retrieval-часть чат-бота

**База данных реплик** включает векторизованные при помощи модели [обученного_энкодера](https://huggingface.co/Shakhovak/chatbot_sentence-transformer) скрипты, включающие контекст и вопрос. Детальное описание процесса обучения я приводила в предыдущем задании. Здесь же просто воспользуюсь готовой моделью из моего репозитория на Hugging Face ([ссылка](https://huggingface.co/Shakhovak/chatbot_sentence-transformer)).

Реплика из базы данных, вопрос и контекст который максимально похожи на тещий запрос в обработке чат-бота (top-1 для ускорения работы), будет подаваться на вход генеративной языковой модели как часть контекста при реализации стратегии RAG (retrieval-augmented generation) для того, чтобы немного добавить фактов и деталей из сценария сериала в диалоге с пользователем.

Для подачи retrieval-реплик в генеративную модель я установила порог в **0.89**. Если топовая реплика имеют косинусную близость с контекстом + вопросом из диалога пользователя, то она не подается в генеративную модель. Если больше, то подается.

Retrieval-реплика включается в контекст перед основным контекстом: ```"context:" + rag_answer + "".join(context) + "</s>" + "question: " + question```. В ходе экспериментов я пробовала различные варианты подачи этой реплики:
- если она добавляется в конце после вопроса, то очень влияет на ответ и модель отвечает уже на эту реплику.
- если отделяется от контекста токеном-сепаратором, то практически не влияет на генарацию
- интереснее всего получаются генерации, если реплика ставится внутри контекста (с сепаратором-токеном или без). Генерации уже не пытаюся ответить на эту реплику, но периодически (не всегда!!!) учитывают факты и стилистику из реплики

### Generative-часть чат-бота
Основной частью чат-бота является генеративная модель. Так как мне удалось получить довольно значительное количество исходных данных, то можно попробовать дообучить небольшие модели семейства T5, которые известны своей универсальностью. В качестве основной для своего эксперимента я выбрала модель ```google/flan-t5-base``` на 248 млн. параметров (детально про модель можно посмотреть [здесь](https://huggingface.co/google/flan-t5-base) в репозитории Hugging Face). Обучаю я модель в облаке в ВМ с иcпользванием A100. Изначально я обучала модель 3 эпохи, но при генерации было много повторений, поэтому в финальной версии я увеличила кол-во эпох обучения до 5. Ноутбук с обучением можно посмотреть [здесь]().

В модель подаюся данные контекста и вопроса в качестве фичей, ответ рассатривается в качестве target. При этом вопрос и контекст соединяются в промпт типа ``` "context: " + контекст + "</s>" + 'question: ' + вопрос ```.

#### Оценка обучения
Для логирования результатов обучения я использовала wandb, репозиторий с мтериками в публичном доступе - детали можно посмотреть по [ссылке](https://wandb.ai/shakhova/generative_models_chat/workspace?nw=nwuserkatya_shakhova).

Во время обучения логировались стандартые метрики, как train/eval loss. Дополнительно я добавила автоматические метрики для сравнения похожести генерируемых текстов ответа с target-ответом из сценария. Использование автоматических метрик для оценки генерации довольно часто критикуется и не может рассматриваться отдельно, в моем случае я использовала их как некое направление, чтобы понять, нужно ли учить модель еще дольше. Данные метрики конечно же нужно дополнять дополнительными human-based оценками уже на этапе подбора стратегии генерации. В качестве автоматических метрик я воспользовалась библиотекой ```evaluate``` на Hugging Face и выбрала из нее метрики [**rouge**](https://huggingface.co/spaces/evaluate-metric/rouge) и [**bertscore**](https://huggingface.co/spaces/evaluate-metric/bertscore), которые представляют из себя пакеты метрик, включающие:

- **rouge 1** - соответствие между сгенерированным текстом и таргетом на основе unigram (чем выше, тем больше похожи тексты)
- **rouge 2** - соответствие между сгенерированным текстом и таргетом на основе 2-gram (чем выше, тем больше похожи тексты)
- **rouge L** - измеряет соответствие самой долгой последовательности между сгенерированным текстом и таргетом (чем выше, тем больше похожи тексты)
- **rouge average generated length** - средняя длина генерируемого текста
- **bertscore recall** - косинусная близость между векторами таргета и генерируемого, оценивемого с помощью эмбедингов модели bert (чем ближе к 1, тем более похожие тексты)
- **bertscore precision** - косинусная близость между векторами генерируемого текста и таргета, оценивемого с помощью эмбедингов модели bert (чем ближе к 1, тем более похожие тексты)
- **bertscore f1** - f1 для предыдущих метрик (чем ближе к 1, тем более похожие тексты)

Ниже принт-скрины из wandb с графиками изменения метрик в процессе обучения:

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/3acbe80e-8f1f-4455-b315-029cb1c32e71)

Продолжение:

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/7850fa68-a6ba-4505-829f-0c5cdf6d5e8e)

Из приведенных графиков видно, что у модели есть еще потенциал для fine-tune, так как продолжают уменьшаться eval и train loss. График eval loss выглядит неплохо - хорошая пологая кривая. По train loss - не совсем ожидаемый наклон уменьшения, есть вероятность, что может не совсем верно подобрано изменение learning rate. Я использовала warm-up ration + weight-decay во время обучения  (см. график ниже).

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/40280326-e463-4023-ada0-a0a3cf27a763)

Несмотря на дальнейший потенциал обучения я остановила обучения на 5 эпохах, так как метрики похожести текстов перестали скачкообразно меняться и вышли на стабильные, хотя и немного растущие значения.

#### Подбор стратегии генерации

Для определения параметров генерации для чат-бота проведу несколько экспериментов с моделью, меняя параметры генерации. Ноутбук с экспериментами можно посмотреть вот [здесь](). В качестве неизменных параметров (после проверки) я выбрала:
```
        do_sample=True - вносим больший элемент рандомности
        max_length=1000 - не ограничиваем генераицт
        repetition_penalty=2.0 - немного недоучена модель, поэтому любит долго и нужно обсждуать одну тему, добавляю штраф на повторение
        top_k=50 - если оставить параметр меньше, то модель вообще плохо следит за репликами пользователя
        no_repeat_ngram_size=2 - продолжение борьбы с недоученностью модели
```

Экспериментировать я буду с парамтерами ```top-p``` и ```temperature``` - оценю, как они влияют на повторяемость и креативность диалога. Оценивать буду генерации по косинусной близости между сгенирированным текстом и таргетом из сценариев. Данные возьму из файла scripts_reworked.pkl, отберу из этого файла рандомную выборку в размере 30 семплов и сравню ответ модели на семпл с таргетным ответом. В качестве bi-encoder возьму свою модель, которую использую для ranking в retrieval-части чат-бота. Дополнительно посмотрю на время генерации. 

В качестве экспериментальных значений возьму следующие параметры:  
- **temperature = 0.2 top_p = 0.1** - ожидаю стандартные тексты, возможно без характеристик героя
- **temperature = 0.5 top_p = 0.5** - ожидаю стандартные тексты, немного больше свободы для генерации у модели
- **temperature = 0.7 top_p = 0.8** - больше креативности, надеюсь герой уже будет проявляться
- **temperature = 1 top_p = 0.95** - возможен уход от контекста

Метрики и сами сгенерированные тексты в качестве артефактов залогирую в wandb в тот же репозиторий, что и метрики при обучении  - детали можно посмотреть по [ссылке](https://wandb.ai/shakhova/generative_models_chat/workspace?nw=nwuserkatya_shakhova).

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/590a8ad5-e951-4bd7-aeda-455e5672d127)

Если смотреть на косинусную близость (см. графики ниже), то видно, что генерации при сочетании temp=1 top_p=0.95 чаще всего похожи на таргетные (показатель косинусной близости реже бывает ниже 0,6), т.е. лучще передают стилистику персонажа. При этом эти генерации чаще всего занимают больше времени.

Посмотрим на сами тексты:

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/250cfa2f-a675-479b-8349-a60be2ad7f64)

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/74b9979b-a50c-4d47-b0ab-7fc220fe9d56)

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/5be62415-ef5f-4deb-9b29-cb532953184b)

![image](https://github.com/shakhovak/chat_bot_katya_HW2/assets/89096305/dc9d55d6-c5ad-4e8b-ad70-c92fac7dee79)


Из сгенерированных текстов очень заметно, что повышение обоих параметров ведет к генерации более интресных и разнообразных текстов. Тексты с низкими параметрами выглядят довольно скучно и ожидаемо не передают характера персонажа. При высоких параметрах остается риск ухода от модели от контекста и придумывания собственных фактов. Интересно, что показатели косинусной близости не сильно отличается, что подверждает сделанные ранее выводы о том, что рассчетные метрики при генерации текста нельзя использовать без оценки генераций человеком.

> [!IMPORTANT]
> Финальные метрики для генерации **temperature=0.95 и top_p=0.9**

## Структура репозитория

```bash
│   README.md - отчет для ДЗ
│   requirements.txt
│   Dockerfile
│   .dockerignore
│   .gitignore
|
│   generate_bot.py - основной файл алгоритма
│   utils.py - вспомогательные функции в т.ч. для предобработки данных
|   app.py - для запуска UI c flask
|
├───train_models - ноутбуки с обучением и оценкой модели
├───templates - оформление веб-интерфейса
│       chat.html
├───static - оформление веб-интерфейса
│       style.css
├───data
│       scripts_reworked.pkl - дополненные данные для обучения модели
│       scripts_vectors.pkl - база данных контекст+воппрос на основе векторов LaBSe
│       scripts.pkl - исходные данные
```

## Веб-сервис
Реализован чат на основе Flask, входной скрипт ```app.py```, который выстраивает графический интерфейс - за основу взят дизайн с [tutorial](https://www.youtube.com/watch?v=70H_7C0kMbI&list=WL&index=4&t=105s)-  создает инстант класса ChatBot, загружает файлы и модели. Также есть Dockerfile, который я использовала, чтобы развернуть сервис на сервере hugging face. 

> [!IMPORTANT]
> Попробовать поговорить с Шелдоном можно по [ссылке](http://158.160.54.0:5000/). Это публичный IP ВМ на Yandex Cloud (ВМ: 60Гб размер дисков, 20Гб RAM, 2CPU)

Хочу обратить внимание, что бесплатное размещение не включает GPU, только CPU, поэтому инференс работает медленнее, чем на локальном компьютере. Кроме того, сервер на Hugging face работает в течение 48 часов после последнего посещения, поэтому при проверке может понадобиться еще раз запустить сервер.

Один из минусов моей реализации - отсутствие сессий и обновления контекста для каждого пользователя/сессии. Я оставила только ограничения по общему размеру контекста (не более 10 предложений).

### Асинхронность на уровне кода Flask-приложения
Начиная с версии 2.0 во Flask добавлены asynchronous route handlers, которые позволяют использовать асинхронный режим на уровне обработки событий самого приложения с помощью ```async ``` ```await ```. Когда запрос поступает в асинхронное представление, то Flask запускает цикл обработки событий в отдельном потоке, запускает там функцию представление, а затем возвращает результат.

В моей реализации у flask-приложения изначально было всего 2 события:
- построение интерфейсв
- получение запроса и генерация ответа от пользователя (здесь не может быть асинхронности, так как нужно сперва получить вопрос, чтобы сгенерировать результат)

  Для демонстрации асинхронности я добавила корутину, которую должна долждаться задача генерации и которая будет выполняться параллельно с ней - это небольшой sleep:

```python
async def sleep():
    await asyncio.sleep(0.1)
    return 0.1

@app.route("/get", methods=["GET", "POST"])
async def chat():
    msg = request.form["msg"]
    input = msg
    await asyncio.gather(sleep(), sleep())
    return get_Chat_response(input)

```
Каждый запрос по-прежнему связывает одну задачу, даже для асинхронных представлений. Положительным моментом является то, что асинхронный код можно запускать в самом представлении, например, для выполнения нескольких одновременных запросов к базе данных и/или HTTP-запросов к внешнему API и т. д. **НО количество запросов, которые веб-приложение может обрабатывать одновременно, останется прежним**. Поэтому переходим к следующему пункту :)

### Многопроцессорность и асинхронность gunicorn
**Gunicorn**  - WSGI (Web-Server Gateway Interface) для UNIX используется для создания многопроцессорности (cоздание нескольких workers) и возможности работать с приложением нескольким пользователем одновременно. Использование ```gevent``` позволяет workers работать в асинхронном режиме и принимать несколько соединений на одного worker. При указании кол-ва соединений на 1 worker можно использовать нуказанное кол-во клонов.

Для запуска такого режима gunicorn нужно прописать в dockerfile: ```CMD ["gunicorn", "--timeout", "1000", "--workers", "2", "--worker-class", "gevent", "--worker-connections" , "100", "app:app", "-b", "0.0.0.0:5000"]```

При запуске image на ВМ будет загружено gunicorn с двумя рабочими процессами, плюс 50 асинхронных gevent процессов на синхронный процесс gunicorn (50 * 2 = 100).

```
admin@bot:~$ sudo docker run -it --name chat -p 5000:5000 --rm shakhovak/hw2_bot
[2024-03-08 17:25:27 +0000] [1] [INFO] Starting gunicorn 21.2.0
[2024-03-08 17:25:27 +0000] [1] [INFO] Listening at: http://0.0.0.0:5000 (1)
# Теперь используется `gevent` 
[2024-03-08 17:25:27 +0000] [1] [INFO] Using worker: gevent
[2024-03-08 17:25:27 +0000] [7] [INFO] Booting worker with pid: 7
[2024-03-08 17:25:27 +0000] [8] [INFO] Booting worker with pid: 8
```
В идеале нужно запускать такое приложение еще и с использованием веб-сервера nginx для более устойчивой работы, но так как это учебный пример, я решила запустить приложение без него.

> [!NOTE]
> Интересное исследование асинхронности приложений на Flask посмотреть [здесь](https://docs-python.ru/packages/veb-frejmvork-flask-python/asinhronnost-veb-prilozhenii-flask/)

<hr>

### Шпаргалка как развернуть docker image в Yandex Cloud
1. Создать ВМ и убедиться, что у нее открыт наружу нужный порт (в случае с ботом - 5000). Машину создала на Debian
2. Установить на ВМ docker enginе. Инструкция вот [здесь](https://docs.docker.com/engine/install/debian/) для debian. Основные команды:
  - удалить потенциальные конфликты  
  - setup docker apt repository
  - установить докер
3. Залогиниться на docker hub ``` sudo docker login ``` и запулить докер образ на ВМ ```sudo docker pull shakhovak/hw2_bot:latest```
4. Запустить образ на ВМ  ```sudo docker run -it --name chat -p 5000:5000 --rm shakhovak/hw2_bot```

<hr>


