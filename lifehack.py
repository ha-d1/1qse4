n = int(input())
s = input("")

if n < 8:
    print(0)
else:
    encrypted_lifehack = []
    for i in range(26):
        news = ""
        for letter in "lifehack":
            news += chr(ord(letter)-i)
        encrypted_lifehack.append(news)
    max_count = 0
    maxk = 0
    k = 0
    while (k<26) and (max_count<(n//8)):
        count = 0
        i = 0
        while i < n-8 :
            if encrypted_lifehack[k] == s[i:i+8]:
                count += 1
                i += 8
        if count > max_count:
            maxk = k
            max_count = count
        k += 1
    if max_count == 0:
        print(0)
    else:
        print(maxk) 
