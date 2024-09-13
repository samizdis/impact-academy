import paramiko
import re

class ShellHandler:

    def __init__(self, hostname, username, key_filename):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname, username=username, key_filename=key_filename)

        channel = self.ssh.invoke_shell()
        self.stdin = channel.makefile('wb')
        self.stdout = channel.makefile('r')

    def __del__(self):
        self.ssh.close()
        
    def pretty_execute(self, cmd):
        shin, shout, sherr = self.execute(cmd)
        output = ''.join(shout)
        error = ''.join(sherr)
        return output if len(error) == 0 else f"""{output}\nError: {error}"""

    def execute(self, cmd):
        """

        :param cmd: the command to be executed on the remote computer
        :examples:  execute('ls')
                    execute('finger')
                    execute('cd folder_name')
        """
        cmd = cmd.strip('\n')
        finish = 'end of stdOUT buffer. finished with exit status'
        echo_cmd = "echo {} $?".format(finish)
        self.stdin.write(cmd + ' ; ' + echo_cmd + '\n')
        shin = self.stdin
        self.stdin.flush()

        shout = []
        sherr = []
        exit_status = 0
        for line in self.stdout:
            if echo_cmd in str(line):
                # up for now filled with shell junk from stdin
                shout = []
            elif finish in str(line):
                # our finish command ends with the exit status
                exit_status = int(str(line).rsplit(maxsplit=1)[1])
                index = str(line).find(finish)
                shout.append(str(line)[:index]) 
                if exit_status:
                    # stderr is combined with stdout.
                    # thus, swap sherr with shout in a case of failure.
                    sherr = list(shout)
                    shout = []
                break
            else:
                # get rid of 'coloring and formatting' special characters
                formatted_line = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', line).replace('\b', '').replace('\r', '')
                shout.append(formatted_line)

        # first and last lines of shout/sherr contain a prompt
        # if shout and echo_cmd in shout[-1]:
        #     shout.pop()
        # if shout and cmd in shout[0]:
        #     shout.pop(0)
        # if sherr and echo_cmd in sherr[-1]:
        #     sherr.pop()
        # if sherr and cmd in sherr[0]:
        #     sherr.pop(0)

        return shin, shout, sherr