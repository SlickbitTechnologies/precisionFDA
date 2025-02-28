#!/bin/bash




sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common

sudo apt install -y python3.9 python3.9-venv python3.9-dev

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
sudo update-alternatives --set python3 /usr/bin/python3.9


git clone https://github.com/SlickbitTechnologies/precisionFDA.git
cd precisionFDA

pip install -r requirements.txt

ollama pull llama3.2:1b
ollama pull nomic-embed-text

streamlit run main.py --server.port=8080 & 
cd ..

if [[ $? -eq 99 ]]; then 
    if [[ $iteration -eq $limit ]]; then 
        echo "Maximum times allowed - Error"
        exit 2
    else
        ((iteration++))  # Increment iteration
        sleep "$seconds_in_wait"
    fi 
else
    iteration=$((limit + 1))  # Ensure loop exits
fi

done
