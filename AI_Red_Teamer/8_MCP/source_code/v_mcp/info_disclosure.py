import asyncio
from fastmcp import Client, FastMCP

client = Client("http://83.136.254.84:39313/mcp/")

async def main():
    async with client:
        resources = await client.list_resources()
        resource_templates = await client.list_resource_templates()
        tools = await client.list_tools()

        print("Resources:")
        for resource in resources:
            print('***')
            print(resource.name)
            print(resource.description.strip())

            

        print("-"*50)
        print("Resource Templates:")
        for resource_template in resource_templates:
            print('***')
            print(resource_template.uriTemplate)
            print(resource_template.description.strip())


        print("-"*50)
        print("Tools:")
        for tool in tools:
            print('***')
            params = list(tool.inputSchema.get('properties').keys())
            print(f"{tool.name}({','.join(params)})")
            print(tool.description.strip())
        
        try:
            result_object1 = await client.read_resource("quantity://banana")
            print(result_object1[0].text)
        
        except Exception as e:
            print(f"[-] {e}")
            
        try:
            result_object = await client.read_resource("resource://logs")
            print(result_object[0].text)
        except Exception as e:
            print(f"[-] {e}")


asyncio.run(main())