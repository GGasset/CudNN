def main():
    files = ['NN.h', 'NN_constructor.h', 'costs.cuh', 'neuron_operations.cuh', 'data_type.h', 'evolution_info.h']
    for file_name in files:
        file_contents = ''
        
        with open('./' + file_name, 'r') as file:
            append : bool = True
            for line in file:
                if '#ifdef INCLUDE_BACKEND' in line:
                    append = False
                if append:
                    file_contents += line
                if '#endif' in line:
                    append = True
        
        with open('./Include/' + file_name, 'w') as file:
            file.write(file_contents)

if __name__ == '__main__':
    main()
