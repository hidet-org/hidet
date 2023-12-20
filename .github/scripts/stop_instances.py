import os
import time
import subprocess

def run_command(cmd):
    print("Running command: " + " ".join(cmd))
    output = subprocess.run(cmd, capture_output=True, text=True)
    print(output.stdout)
    print(output.stderr)
    return output

if __name__ == '__main__':
    instances = os.environ.get('STARTED_INSTANCES').replace(' ', '')
    instances = [s for s in instances.split(';') if s]
    # Stop all instances
    for instance in instances:
        ids = [s for s in instance.split(',') if s]
        cloud_provider_id = int(ids[0])
        instance_id = ids[1]
        if cloud_provider_id == 1: # AWS
            cmd = ['aws', 'ec2', 'stop-instances', '--instance-ids', instance_id]
        else:
            raise ValueError(f'Unknown cloud provider id: {cloud_provider_id}')
        output = run_command(cmd)
        if output.returncode != 0:
            raise RuntimeError(f'Failed to stop instance {instance_id} on cloud provider {cloud_provider_id}.')

    # Wait until all instances are stopped
    for instance in instances:
        ids = [s for s in instance.split(',') if s]
        cloud_provider_id = int(ids[0])
        instance_id = ids[1]
        stopped = False
        while not stopped:
            time.sleep(5)
            if cloud_provider_id == 1: # AWS
                cmd = ['aws', 'ec2', 'describe-instance-status', '--instance-ids', instance_id]
                output = run_command(cmd)
                if output.returncode != 0:
                    raise RuntimeError(f'Failed to check status for {instance_id} on cloud provider {cloud_provider_id}.')
                # A stopped instance will give empty instance status.
                # An instance still running would contain its id in the status.
                if instance_id not in output.stdout:
                    stopped = True
            else:
                raise ValueError(f'Unknown cloud provider id: {cloud_provider_id}')