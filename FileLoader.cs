using UnityEngine;
using System.Collections;
using System.IO;
using SimpleFileBrowser;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class FileLoader : MonoBehaviour
{
	// Warning: paths returned by FileBrowser dialogs do not contain a trailing '\' character
	// Warning: FileBrowser can only show 1 dialog at a time

	public Sprite sp,sl,sprite;
	public SpriteRenderer sr;
	int w, h,bw,bh;
	float ppu;
	public string path;
	int count = 0;

	Scene currentScene;
	void Start()
	{
		GameObject gm = GameObject.Find("Button");
		gm.GetComponent<Button>().interactable = false;
		sr = gameObject.GetComponent<SpriteRenderer>();
		currentScene = SceneManager.GetActiveScene();
		string sceneName = currentScene.name;

		if (sceneName == "Start")
		{
			sr.enabled = false;
		}
		else
        {
			sr.sprite = sprite;
			sr.enabled = true;
        }


		Debug.Log("ok");

	}

	public void LoadImage()
	{
		FileBrowser.SetFilters(false, new FileBrowser.Filter("Images", ".png", ".jpg", ".jpeg"));
		FileBrowser.ShowLoadDialog((paths) =>
		{
		// Creating a Texture2D from picked image
		Texture2D texture = new Texture2D(2, 2);
		texture.LoadImage(File.ReadAllBytes(paths[0]));
			path = paths[0];

			// Creating a Sprite from Texture2D
		double ratio=texture.width / texture.height;
			bw = texture.width;
			bh = texture.height;
			w = bw*100;
			h = bh*100;
			Debug.Log("" + texture.width + " " + texture.height + "ppu:" + ppu);
			//if(texture.width>1400)
			//  {
			//	w = 1400;
			//  h = (int)(1400 * ratio);
			//    }
			//if (texture.height>700)
			//  {
			//	h = 700;
			//	w = (int)(700 * ratio);
			//  }
			ppu = 100f;
			
		while (w>(1400) || h>(700))
			{
				ppu =ppu+1;
				w =(int) (bw / (ppu/100));
				h =(int) (bh / (ppu/100));
				Debug.Log("" + w + " " + h + "ppu:" + ppu);
			}

		
		sprite = Sprite.Create(texture, new Rect(0f, 0f, texture.width, texture.height), new Vector2(0.5f, 0.5f),ppu);
			Debug.Log (""+texture.width+" "+texture.height+"ppu:"+ppu);

			GameObject gm = GameObject.Find("Button");
			  gm.GetComponent<Button>().interactable = true;

			// Calling AssignSpriteToObject with that Sprite
			AssignSpriteToObject(sprite);
		}, null, FileBrowser.PickMode.Files);
	}

	public void LoadImage2()
	{
		count++;

		//FileBrowser.SetFilters(false, new FileBrowser.Filter("Images", ".png", ".jpg", ".jpeg"));
		//FileBrowser.ShowLoadDialog((paths) =>
		
			// Creating a Texture2D from picked image
			Texture2D texture = new Texture2D(2, 2);
		Debug.Log("jshklj");
		if(count==1)
			texture.LoadImage(File.ReadAllBytes(Application.dataPath + "/car1.png"));
		if (count == 2)
			texture.LoadImage(File.ReadAllBytes(Application.dataPath + "/cheetah1.png"));
		if (count == 3)
			texture.LoadImage(File.ReadAllBytes(Application.dataPath + "/mount1.png"));
		if (count == 4)
			texture.LoadImage(File.ReadAllBytes(Application.dataPath + "/tree1.png"));

		// Creating a Sprite from Texture2D
		double ratio = texture.width / texture.height;
			bw = texture.width;
			bh = texture.height;
			w = bw * 100;
			h = bh * 100;
			Debug.Log("" + texture.width + " " + texture.height + "ppu:" + ppu);
			//if(texture.width>1400)
			//  {
			//	w = 1400;
			//  h = (int)(1400 * ratio);
			//    }
			//if (texture.height>700)
			//  {
			//	h = 700;
			//	w = (int)(700 * ratio);
			//  }
			ppu = 100f;

			while (w > (1400) || h > (700))
			{
				ppu = ppu + 1;
				w = (int)(bw / (ppu / 100));
				h = (int)(bh / (ppu / 100));
				Debug.Log("" + w + " " + h + "ppu:" + ppu);
			}


			sprite = Sprite.Create(texture, new Rect(0f, 0f, texture.width, texture.height), new Vector2(0.5f, 0.5f), ppu);
			Debug.Log("" + texture.width + " " + texture.height + "ppu:" + ppu);

			GameObject gm = GameObject.Find("Button");
			gm.GetComponent<Button>().interactable = true;

			// Calling AssignSpriteToObject with that Sprite
			AssignSpriteToObject(sprite);
		// null, FileBrowser.PickMode.Files);
	}
	public void AssignSpriteToObject(Sprite sprite)
	{
		sr.sprite = sprite;
		sr.enabled = true;
	}

	public Sprite getSprite()
    {
		return sprite;
    }

	public string getPath()
	{
		return path;
	}
}