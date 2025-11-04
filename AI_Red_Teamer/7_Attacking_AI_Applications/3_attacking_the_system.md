# Excessive Data Handling & Insecure Storage

Security vulnerabilities related to the insecure storage of data can result in unauthorized access.

## Insecure Data Storage

ML applications frequently store training data, intermediate results, logs, or user inputs for retraining or analytics purposes. If these data stores are not adequately secured, they can be prime targets for attackers. Insecure data storage can result from a lack of encryption or broken access control. Compromised data repositories may expose sensitive information, enabling downstream attacks like identity theft, profiling, or unauthorized model training. 

### Example 
In the example there is a chatbot that helps place and suggest orders to users based on medical conditions. We see by interacting with the bot and asking things such as "what information do you need to place an order" the bot asks for a credit card number.

The next step is to conduct directory brute-forcing: 
```
gobuster dir -u http://<SERVER_IP>:<PORT>/ -w /opt/SecLists/Discovery/Web-Content/raft-small-words.txt -x .db,.txt,.html
```

This reveals a database file which we then attempt to download. 

```
wget http://<SERVER_IP>:<PORT>/<FILE>.db

```

We see it was successful and we can use the file command to see the file contents 
```
file storage.db
```

Results: 
```
storage.db: ASCII text, with very long lines (533)
```

by looking at the file contents we identify IP addresses and credit card details of users. 


## Mitigations
1. Principle of Data minimizations (collecting only the data vital for the ML task.)
2. Use Data anonymization or differential privacy
4. Follow best practices for secure data storage
    - access control mechanisms
    - encryption
    - data retention policiates
5. Avoid plaintext data by using Homomorphic Encryption (HE).

*HE allows computations to be performed directly on encrypted data, producing encrypted results that, when decrypted, match the outcome of operations performed on the raw data.*