# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model> <bit_width>"
    exit 1
fi

MODEL=$1
BIT_WIDTH=$2

echo "Starting Execution of Training with bit_width "${BIT_WIDTH} 
python ptq.py --model ${MODEL} --bit_width ${BIT_WIDTH}