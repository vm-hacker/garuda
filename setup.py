import os
import getpass

def install_dependencies():
    print("Installing requirements(dependencies)...")
    os.system("sudo apt-get install python3-pip -y")
    os.system("pip install --upgrade pip")
    os.system("pip install -r dependencies.txt")


def main():
    username = getpass.getuser()
    print("Hi,",username)
    hostname = os.uname()[1]
    print(f"Setting up Garuda on: {hostname}")
    for i in range(1):
        install_dependencies()
        print("Deploying Garuda...")
        os.system("python3 bot.py")
        print("Done!")

if __name__ == "__main__":
    main()