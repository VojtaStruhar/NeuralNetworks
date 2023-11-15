from random import randint

def generate_xor(amount: int,inputs_file: str, output_file: str) -> None:
    inp = open(inputs_file, 'w')
    out = open(output_file, 'w')

    for i in range(amount):
        a = randint(0, 1)
        b = randint(0, 1)
        
        inp.write(str(a) + ',' + str(b) + '\n')
        out.write(str(a ^ b) + '\n')

    inp.close()
    out.close()

if __name__ == '__main__':
    generate_xor(60_000, 'data/xor_train_inputs.csv', 'data/xor_train_outputs.csv')
    generate_xor(10_000, 'data/xor_test_inputs.csv', 'data/xor_test_outputs.csv')