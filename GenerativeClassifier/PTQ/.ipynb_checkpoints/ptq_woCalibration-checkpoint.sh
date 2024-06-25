# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <bit_width>"
    exit 1
fi

BIT_WIDTH=$1

echo "Starting Execution of Training with bit_width "${BIT_WIDTH} 
python ptq_woCalibration.py --bit_width ${BIT_WIDTH}