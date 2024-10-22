import argparse
import os
import hnswlib
import numpy as np
import pandas
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def read_fbin(filename):
    with open(filename, 'rb') as f:
        num_elements = np.fromfile(f, dtype=np.int32, count=1)[0]
        dimension = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(num_elements, dimension)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='datasets/base.10M.fbin')
    parser.add_argument('--result', type=str, default='components-test-results.csv')
    parser.add_argument('--m', type=int, default=16)
    parser.add_argument('--ef', type=int, default=64)
    args = parser.parse_args()
    
    print("Загружаем данные из " + args.dataset)
    data = read_fbin(args.dataset)
    num_elements, dimension = data.shape
    
    print("Инициализируем индекс")
    num_threads = os.cpu_count()
    index = hnswlib.Index(space='l2', dim=dimension)
    index.init_index(max_elements=num_elements, ef_construction=args.ef, M=args.m)
    
    batch_size = 10000  # Вы можете настроить размер батча в соответствии с вашей памятью
    for i in tqdm(range(0, num_elements, batch_size), desc="Добавление элементов"):
        end = min(i + batch_size, num_elements)
        index.add_items(data[i:end], num_threads=num_threads)
    
    print("Выполняем k-NN запросы")
    labels, distances = index.knn_query(data, k=args.m, num_threads=num_threads)
    
    print("Строим разреженную матрицу смежности")
    row_indices = np.repeat(np.arange(num_elements), args.m)
    col_indices = labels.flatten()
    data_values = np.ones(len(row_indices), dtype=np.bool_)
    
    adjacency_matrix = coo_matrix((data_values, (row_indices, col_indices)), shape=(num_elements, num_elements))
    
    # Делаем матрицу симметричной для неориентированного графа
    adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose()
    adjacency_matrix.data = np.ones_like(adjacency_matrix.data, dtype=np.bool_)  # Устанавливаем все значения в 1
    
    print("Считаем компоненты связности")
    num_components, labels = connected_components(csgraph=adjacency_matrix, directed=False, return_labels=True)
    
    print("Количество компонент связности: " + str(num_components))
    
    print("Сохраняем результат с помощью pandas")
    result_data = {
        'm': [args.m],
        'ef': [args.ef],
        'num elements': [num_elements],
        'dimension': [dimension],
        'connected components': [num_components]
    }
    df = pandas.DataFrame(result_data)
    
    if not os.path.isfile(args.result):
        df.to_csv(args.result, index=False)
    else:
        df.to_csv(args.result, mode='a', header=False, index=False)

if __name__ == '__main__':
    main()