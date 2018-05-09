# liborigin

Библиотека для чтения файлов типа `OPJ` (Origin).

## Сборка библиотек liborigin

Скачиваем репозиторий с github:

```bash
git clone https://github.com/Saluev/python-liborigin2.git
```

Собираем библиотеку:

```bash
mkdir build
cd build
cmake ..
make
```

Библиотека - файл: `liborigin.<platform>.so` - скопировать в рабочую директорию.

## Документация

Собираем документацию:

```bash
doxygen Doxyfile
```

Документация появится в папке html, стартовая страница index.html.


## Использование liborigin

```python
import liborigin
```

Открываем файл *.opj:

```python
dict_content = liborigin.parseOriginFile(file_path)
```

Тип результата - словарь (dict).

Поля в словаре:

* functions
* matrices
* spreads
* graphs
* notes

Нам требуются таблицы с данными. Они лежат в списке в поле "spreads"

Тип таблицы: `liborigin.SpreadSheet`

Поля таблицы `liborigin.SpreadSheet`:

* name <bytes>
* columns <list> of <liborigin.SpreadColumn>
* sheets <int>
* state <int>
* title <int>

Поля `liborigin.SpreadColumn`:

* name <bytes>
* comment <bytes>
* index <int>
* data <list> of <float> - собственно данные, которые нам нужны
