import subprocess

# 调用 PowerShell 执行指令
cmd = r'cd D:\tmp; echo "hello world" > a.txt'

subprocess.run(
    ["powershell", "-Command", cmd],
    check=True,
    shell=False
)
