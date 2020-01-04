def pow_mod(a,n,mod):
    re=1
    while n :
        if n&1 :
            re=(re*a)%mod
        n >>=1
        a=(a*a)%mod
    return re %mod

if __name__ == "__main__":
    print("helloWorld!")
    t=input()
    
    for i in range(0,int(t)):
        str1=input()
        list1=str1.split(" ")
        a=int(list1[0])
        b=int(list1[1])
        m=int(list1[2])
        if b==0 or a==1:
            print(1)
            continue
        #print(a," ",b," ",m)
        ans=1
        ''''''
        for j in range(0,b-1):
            ans=pow(a,ans)
        print(pow_mod(a,ans,m))
