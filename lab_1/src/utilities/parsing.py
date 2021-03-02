def matrix_parsing(rows, cols):
    matrix = []

    if rows == 0 or cols == 0:
        print('ошибка')
        return

    try:
        for row in range(1, rows + 1):
            elements = input(f'Введите элементы для {row}-ой строки через пробел: ')
            subarray = list(map(float, elements.split()))
            if len(subarray) == cols:
                matrix.append(subarray)
            else:
                raise ValueError('Превышение по количеству столбцов')
        return matrix
    except:
        raise ValueError('Ошибка')
