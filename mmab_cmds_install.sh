#! /bin/bash
cat>/tmp/.mmab_cmds<<EOF
import os, time
import glob
import tempfile
import shutil
import argparse
import subprocess

def query_ok(msg):
    global args
    if args.yes:
        return True
    r = input(msg+"(y/n):")
    if r == "y":
        return True
    else:
        return False

def main():
    global args
    # 创建解析器
    parser = argparse.ArgumentParser(prog="mmab", description="mmab Command Line Tool")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # 添加子命令 'install'
    parser_install = subparsers.add_parser("install", help="安装和更新mmab")
    parser_install.add_argument("-p", "--path", type=str, default="/root/corespace", help="mmab安装路径, 不指定时默认为/root/corespace")
    parser_install.add_argument("-u", "--update", action='store_true', help="是否更新mmab")
    parser_install.add_argument("-y", "--yes", action='store_true', help="不询问, 直接执行")

    # 添加子命令 'example'
    parser_example = subparsers.add_parser("example", help="安装mmab的代码示例")
    parser_example.add_argument("-n", "--name", type=str, help="指定需要安装的示例名称")
    parser_example.add_argument("-ls", "--list", action='store_true', help="列出可以安装的示例名称")
    parser_example.add_argument("-p", "--path", type=str, default="/root/corespace/mmab", help="mmab已安装路径, 不指定时默认为/root/corespace/mmab")
    parser_example.add_argument("-y", "--yes", action='store_true', help="不询问, 直接执行")

    # 添加子命令 'dataset'
    parser_dataset = subparsers.add_parser("dataset", help="安装指定的数据集")
    parser_dataset.add_argument("-n", "--name", type=str, help="指定需要安装的数据集名称")
    parser_dataset.add_argument("-ls", "--list", action='store_true', help="列出可以安装的数据集名称")
    parser_dataset.add_argument("-p", "--path", type=str, default="/root/workspace/datasets", help="数据集安装目录, 不指定时默认为/root/workspace/datasets")
    parser_dataset.add_argument("-y", "--yes", action='store_true', help="不询问, 直接执行")

    # 解析命令行参数
    args = parser.parse_args()

    if args.command == "install":
        handle_install(args)
    elif args.command == "example":
        handle_example(args, parser_example)
    elif args.command == "dataset":
        handle_dataset(args, parser_dataset)
    else:
        parser.print_help()

def handle_install(args):
    temp_dir = os.path.join("/tmp", "mmab")
    install_path = os.path.abspath(os.path.join(args.path, "mmab"))
    if os.path.isdir(install_path):
        if args.update:
            if not query_ok(f"{install_path}已经存在, 执行更新将覆盖mmab库中更改过的源文件, 是否继续进行更新?"):
                return
        else:
            if not query_ok(f"{install_path}已经存在, 是否继续进行安装?"):
                return
    if download_mmab(temp_dir):
        move_and_overwrite(temp_dir, install_path)
        print(f"mmab已成功下载到{install_path}")
        print(f"注册mmab pip......")
        register_mmab_pip(install_path)
        print(f"重新编译mmdeploy sdk......")
        time.sleep(5)
        build_dir = "/root/corespace/mmdeploy/build"
        copy_file_and_overwrite(
            os.path.join(install_path, "mmab/mmdeploy/instance_segmentation.cpp"),
            "/root/corespace/mmdeploy/csrc/mmdeploy/codebase/mmdet")
        if os.system(f"cd {build_dir} && make -j") == 0:
            print(f"编译mmdeploy sdk成功")
            print("\nmmab已成功安装, 如需查看运行示例, 执行命令:\nmmab example --list")
            print("如需安装运行示例, 执行命令:\nmmab example --name {name}")
        else:
            print(f"编译mmdeploy sdk可能失败, 请仔细查看输出")
    else:
        print("下载mmab失败")

def download_mmab(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)  # 清理临时目录
    os.makedirs(temp_dir)

    repo_url = "https://github.com/Joyphy/mmab.git"  # 更换为实际的仓库URL
    return shell_run(f"git clone {repo_url} {temp_dir}")

def register_mmab_pip(install_path):
    time.sleep(2)
    pth_file = '/opt/conda/lib/python3.10/site-packages/easy-install.pth'
    if os.path.exists(pth_file):
        with open(pth_file, 'r+') as file:
            lines = file.readlines()
            if any(install_path in line.strip() for line in lines):
                print(f"{install_path}已经注册在pip中")
            else:
                file.write(install_path + '\n')
                print(f"注册{install_path}到pip成功")
    else:
        with open(pth_file, 'w') as file:
            file.write(install_path + '\n')
        print(f"创建{pth_file}并注册{install_path}到pip成功")

def shell_run(cmd) -> bool:
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 实时输出标准输出和标准错误
    while True:
        output = process.stdout.readline()
        if output:
            print(output.strip())
        
        # 检查子进程是否已终止
        if output == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        print(f"\"{cmd}\"命令执行错误: 返回码 {process.returncode}")
        return False

    return True

def move_and_overwrite(src_dir, tar_dir):
    root_dir = src_dir
    for src_dir, dirs, files in os.walk(root_dir):
        dst_dir = src_dir.replace(root_dir, tar_dir, 1)
        os.makedirs(dst_dir, exist_ok=True)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.move(src_file, dst_dir)
    shutil.rmtree(root_dir)

def copy_file_and_overwrite(src, tar_dir):
    for src_file in glob.glob(src, recursive=False):
        os.makedirs(tar_dir, exist_ok=True)
        dst_file = os.path.join(tar_dir, os.path.basename(src_file))
        if os.path.exists(dst_file):
            os.remove(dst_file)
        shutil.copyfile(src_file, dst_file)

def handle_example(args, parser):
    if args.list:
        list_example_print()
    elif args.name:
        name = args.name
        install_path = os.path.abspath(args.path)
        if not os.path.isdir(install_path):
            print("mmab安装路径%s不存在\n请执行安装命令: mmab install\n或请重新指定安装路径: mmab example -n {name} -p {mmab install path}" % install_path)
            return
        create_example_dir(name, install_path)
        cmds_dict = {
            "patchcore": "mmab dataset -n bottle -p /root/workspace/datasets"
        }
        print(f"\n示例运行一般需要数据集, 如还未安装, {name}示例请执行:")
        print(cmds_dict[name])
        print(f"查看其他可安装数据集, 请执行:")
        print("mmab dataset -ls")
    else:
        parser.print_help()

def create_example_dir(name, install_path):
    tar_dir = f"/root/workspace/mmab_{name}_examples"
    if os.path.isdir(tar_dir):
        if not query_ok(f"{tar_dir}目录已经存在, 继续安装将覆盖指定文件?"):
            return
    copy_file_and_overwrite(os.path.join(install_path, f"examples/{name}/*.*"), tar_dir)
    copy_file_and_overwrite(os.path.join(install_path, f"configs/{name}_*.py"), os.path.join(tar_dir, "configs"))
    os.makedirs(os.path.join(tar_dir, "workdirs"), exist_ok=True)
    print(f"创建示例完成, 已安装到{tar_dir}中")
    
def list_example_print():
    print("<示例name>\t\t<示例说明>")
    print("-"*80)
    print("patchcore\t\tpatchcore模型运行示例")
    print("-"*80)

def handle_dataset(args, parser):
    if args.list:
        list_dataset_print()
    elif args.name:
        name = args.name
        install_path = os.path.join(os.path.abspath(args.path), name)
        if os.path.isdir(install_path):
            if not query_ok(f"{install_path}已存在, 继续执行将覆盖该数据集之前的更改?"):
                return
        download_dataset(name, install_path)
    else:
        parser.print_help()

def download_dataset(name, install_path):
    dataset_http_dict = {
        "bottle": "http://10.10.1.24:8888/39/d/@WEBDOC/datasets/MVTec/bottle.tar.xz"
    }
    os.makedirs(install_path, exist_ok=True)
    tar_name = os.path.basename(dataset_http_dict[name])
    temp_dir = tempfile.mkdtemp()
    tar_file = os.path.join(temp_dir, tar_name)

    if os.system(
        f"wget -O {tar_file} {dataset_http_dict[name]} && "
        f"tar -xf {tar_file} -C {install_path} --overwrite --strip-components=1"
    ) == 0:
        print(f"数据集成功安装到{install_path}")
    else:
        print(f"数据集安装可能失败, 请仔细查看输出")
    os.system(f"rm -rf {temp_dir}")

def list_dataset_print():
    print("<datast_name>\t\t<dataset说明>")
    print("-"*80)
    print("bottle\t\tMVTec bottle数据集, 训练集好图209张, 测试集好图20张, 缺陷图63张")
    print("-"*80)

if __name__ == "__main__":
    main()
EOF

if grep -q "alias mmab=" ~/.bashrc; then
    sed -i '/alias mmab=/c\alias mmab='\''python3 /tmp/.mmab_cmds'\''' ~/.bashrc
    echo "Alias 'mmab' updated in ~/.bashrc."
else
    echo "alias mmab='python3 /tmp/.mmab_cmds'" >> ~/.bashrc
    echo "Alias 'mmab' added to ~/.bashrc."
fi

source ~/.bashrc