import argparse
from clipper_admin import ClipperConnection, KubernetesContainerManager

manager = KubernetesContainerManager(
    kubernetes_proxy_addr='127.0.0.1:8001', namespace='mdt')
clipper_conn = ClipperConnection(manager)


def deploy():
    clipper_conn.start_clipper()


def undeploy():
    clipper_conn.stop_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy clippers')
    parser.add_argument('--op', help='opertations support d|u|r')
    args = parser.parse_args()
    
    if args.op == 'd':
        deploy()
    elif args.op == 'u':
        undeploy()
    else:
        print("unsupported operation {}".format(args.op))
