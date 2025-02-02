{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Dog Breed Identification:\n",
    "This is the project from Kaggle's dog-breed-identification competition.\n",
    "The data's are being used here from Kaggle's site provided for this competition. \n",
    "Downloded from \n",
    "https://www.kaggle.com/c/dog-breed-identification/data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Lib imported successfully\n"
     ]
    }
   ],
   "source": [
    "### Import all relevent lib\n",
    "import csv\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "import os, os.path\n",
    "import glob\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "### Dir\n",
    "LABELS = 'data/labels.csv'\n",
    "DATA_FILE = 'data/data.csv'\n",
    "TRAIN_IMAGE_DIR = 'data/train/'\n",
    "TEST_IMAGE_DIR = 'data/test/'\n",
    "IMAGE_SIZE = 32,32\n",
    "CHANNEL = 3\n",
    "SIZE = 32\n",
    "PIXEL_DEPTH = 255\n",
    "# Model save path\n",
    "save_model_path = './dog_breed_recognition'\n",
    "\n",
    "print ('Lib imported successfully')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_image_in_array( indirname ) :\n",
    "    data = []\n",
    "    for root, dirs, fileList in os.walk(indirname, topdown=True):\n",
    "         for fname in fileList:\n",
    "#            for i in range(5):\n",
    "#                if count > 5 :\n",
    "#                    break\n",
    "            f = os.path.join(root,fname)\n",
    "            img = Image.open(f)\n",
    "            img.load()\n",
    "            img = img.resize(IMAGE_SIZE, Image.ANTIALIAS)\n",
    "            tmp = np.asarray( img, dtype=\"int32\" )\n",
    "            data.append(tmp)\n",
    "#            display(img)\n",
    "#            display(len(data))\n",
    "            \n",
    "    data = np.array(data)       \n",
    "    return data \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def save_feature_in_csv(outfile, indata):\n",
    "    with open(outfile, 'w', newline='') as csvFile:\n",
    "        writer = csv.writer(csvFile, delimiter=',', quotechar='\"')\n",
    "        writer.writerow([\"Index\" , \"Feature\"]) \n",
    "    with open(outfile, 'a', newline='') as csvFile:\n",
    "        writer = csv.writer(csvFile, delimiter=',', quotechar='\"')\n",
    "        for i in range(indata.shape[0]):\n",
    "            a = indata[i]\n",
    "            writer.writerow([i+1,a])\n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def label_encoding(labels):\n",
    "    # integer encode\n",
    "    label_encoder = LabelEncoder()\n",
    "    integer_encoded = label_encoder.fit_transform(labels)\n",
    "\n",
    "    # binary encode\n",
    "    onehot_encoder = OneHotEncoder(sparse=False)\n",
    "    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)\n",
    "    onehot_label_encoded = onehot_encoder.fit_transform(integer_encoded)\n",
    "    \n",
    "    print(onehot_label_encoded[0])\n",
    "    return onehot_label_encoded"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def feature_scaling(values):\n",
    "    # Values are rescaled from [0, 255] down to [-0.5, 0.5]\n",
    "    for i in range(values.shape[0]):\n",
    "        values[i] = (values[i] - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH\n",
    "    return values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def organise_data():\n",
    "    # load the image data in to np array of size (no of image, 32, 32, 3)\n",
    "    data = load_image_in_array(TRAIN_IMAGE_DIR)\n",
    "    #display(data.shape)\n",
    "    \n",
    "    data = np.array(data, dtype=np.int32).flatten()\n",
    "    #display(data.shape)\n",
    "    # reshape the image into [image index, image_size*image_size*channels]\n",
    "    jpgCounter = len(glob.glob1(TRAIN_IMAGE_DIR,\"*.jpg\"))\n",
    "    \n",
    "    data = data.reshape(jpgCounter,SIZE*SIZE*3)\n",
    "    #display(data.shape)\n",
    "    \n",
    "    # save the data in /data/data.csv file\n",
    "    save_feature_in_csv(DATA_FILE, data)\n",
    "    \n",
    "    return data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def get_feature_label():\n",
    "    labels = np.loadtxt(open(LABELS, \"rb\"), dtype=str, delimiter=\",\", skiprows=1, usecols = 1)\n",
    "    features = organise_data()\n",
    "    features = features.reshape(features.shape[0], SIZE, SIZE, CHANNEL)\n",
    "    #features = np.loadtxt(open(\"data/data.csv\", \"rb\"), dtype=int, delimiter=\",\", skiprows=1)\n",
    "    display(labels.shape)\n",
    "    onehot_label_encoded = label_encoding(labels)\n",
    "    scaled_features = feature_scaling(features)\n",
    "    display(onehot_label_encoded.shape)\n",
    "    return scaled_features, onehot_label_encoded"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def get_train_cv_test_data():\n",
    "    features, labels = get_feature_label()\n",
    "    #features = features.reshape(features.shape[0], SIZE, SIZE, CHANNEL)\n",
    "    train_features, test_features, train_labels, test_labels = train_test_split(\n",
    "        features,\n",
    "        labels,\n",
    "        test_size=0.2,\n",
    "        random_state=42)\n",
    "    train_features, valid_features, train_labels, valid_labels = train_test_split(\n",
    "        train_features,\n",
    "        train_labels,\n",
    "        test_size=0.1,\n",
    "        random_state=42)\n",
    "    \n",
    "    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels\n",
    "\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10222,)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(10222, 120)"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_features, train_labels, valid_features, valid_labels, test_features, test_labels = get_train_cv_test_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((7359, 32, 32, 3),\n",
       " (7359, 120),\n",
       " (818, 32, 32, 3),\n",
       " (818, 120),\n",
       " (2045, 32, 32, 3),\n",
       " (2045, 120))"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_features.shape, train_labels.shape, valid_features.shape, valid_labels.shape, test_features.shape, test_labels.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "### CNN Model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1d59734c5c0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "image = train_features[6000]\n",
    "#display(image)\n",
    "#img = np.reshape(image,newshape=(32,32,3))\n",
    "img = img.astype(np.uint8)\n",
    "plt.imshow(img) #load\n",
    "plt.show()  # show the window\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAcAAAAELCAYAAABOClYEAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBo\ndHRwOi8vbWF0cGxvdGxpYi5vcmcvpW3flQAAIABJREFUeJzt3Xm8JFV99/HPl2EZEGEERoEBMoNM\nVNRIcGRxC4pBQBYlkECMAiEhGsT9FSFoICAGeRI3XJ7wCGGJAYGYsIggQQZiwjYgsg3IAAOMbIMz\nDDMODLP8nj/OKW5NU7e7771dfZf6vl+vfnXX6VOnTlVX1a/q1KlqRQRmZmZNs85oV8DMzGw0OACa\nmVkjOQCamVkjOQCamVkjOQCamVkjOQCamVkjOQCamVkjOQCamVkjOQCamVkjrTvaFei3LbbYIqZP\nnz7a1TAzGzduu+22ZyJi6mjXo9caFwCnT5/OnDlzRrsaZmbjhqRHRrsOdXATqJmZNZIDoJmZNZID\noJmZNZIDoJmZNZIDoJmZNZIDoJmZNZIDoJmZNZIDoJmZNZIDoJmZNVLjngQzEm/40lUAzD1l766+\n2/+Mn78s3+XHvrNtOcU4lx/7zhHVtVxOuzKrvhtOHeqodzszjv8xAG/aetOX8veqDu3qUsey6XW9\nR6LdetxN/cbSvJR1W68669+L9Wmw/K37mrG2/Meq2gKgpLOB/YCnI+JNOW0z4IfAdGA+8McRsViS\ngG8C+wLLgSMi4vY8zuHAF3OxX46Ic3P6W4FzgA2BK4FPRUTUNT+dvLBqddvv5z29bMhlFjujHV69\nMbD2Tr4oryqIdqO1nG7z171h3f34khHXobwz6DboVykvm6qDmU7lFb9ft+P0Krh2Shvqciz/JlU7\n2uHMy1Dntd1B43DL7nXQ6LaO5XzFOjbRD6zGqjrPAM8Bvg2cV0o7Drg2Ik6TdFwe/gKwDzAzv3YF\nvgfsmgPmicAsIIDbJF0WEYtznqOBm0gBcG/gJzXOT9fa7SzLyhtC1ThFUC3viIud0XBCfZ0bRFGv\nTsGnV/Uq5n84Bxa9qkOncp5fmX6/Ddeb9NL3rWeuMPR5aBdwhlMeVAfrbsupytdp3Hbfj/Rsrd1B\nVK+021476ZSv3fwPNm/F8uy2DpbUFgAj4gZJ01uSDwT2yJ/PBWaTAuCBwHn5DO4mSVMkbZXzXhMR\niwAkXQPsLWk2sElE3JjTzwM+yCgEwKoNoVgZizM3qF4xix1kr+vQzU58qDvJqvzlDbUqIA/nTGuo\nqpbrSINitwcw3U5vJDulTmcVVeta1XpV3nFW1bcqWBc6HWyV56/bZd9u3R9O4K1a14az7NqdkdVx\nltZu3RjK9Ipyyr9Vrw4OJ7J+XwN8TUQ8ARART0h6dU6fBjxWyrcgp7VLX1CRXknS0aSzRbbbbrth\nV75qo+3VSlZVTtWOZySNvCM5Mi5vqEMNEOVx6jwr7FROu3qXl/9Qf9NOAa78m3X7+7VrPq2a9mB1\nLsopT3ckAbkqaFTNU7fTqPqdh3OwOJwzzqrpVKV1u950UuQdzvowEm4KHdxY6QSjirQYRnqliDgT\nOBNg1qxZPV3lqo68Ou2UhroD6vb6YnmnVKXbja3bYDwUVWUWzYEjMZSdb7vl1O1vcteve9+8Vkx7\n8rovP/uqOvCo88i+03IYS01sw6lLp2Aw1G2kHJhVsVcazm9VHKiWm8mr6ldny0NT9DsAPiVpq3z2\ntxXwdE5fAGxbyrcN8HhO36MlfXZO36Yi/5hSXgFHEkC6bYIabIMY6plftxvOUDbuosyqZqluj4i7\nnd5gZfTizKes6uxqODu8boN4rw9MqoJBt2eKndap4RxsFfXpNO5wWiDaGU7LSJ3BZfS68jVPvwPg\nZcDhwGn5/dJS+ickXUjqBLMkB8mrga9IelXOtxdwfEQskrRU0m7AzcBHgTP6OSOF4aysY2EF78X1\nx34dYXbbHFgVIPrRIaIOxTpSPquoam3opOp37nb8dvn6tQ53G+yqDjbruD5cl0716nYbKBsL+5mx\nrs7bIC4gnb1tIWkBqTfnacBFko4CHgUOydmvJN0CMY90G8SRADnQnQLcmvOdXHSIAT7OwG0QP6GP\nHWC6bUapahLppFcdY3rdO3KkRrIxtuugUZVvsOm266hTpc4AP5yz2fG6QxvJ7Qbd9jCtc9kMp4dl\nMc+9Woc67Rd60XLSRHX2Aj1skK/2rMgbwDGDlHM2cHZF+hzgTSOpY92GslGO1fb6oW48nc64xsJ8\nVtVxvAaXQq8OnEZqqGddI1kfevWbVZXTq6DRbR3Hyu/XNGOlE4z12Eg6CHRS1ZttOMZ70LGRq3Md\nKJfdbStCobz9VDU9D6eHtoPc2OMAOAH0aifSbVfxJurHcuj3DnKsNY1VnZn3ern3a33uxbXn8Xr9\nejxxALQxqc6NfzidQLodZ6IfMIzkYGusLZumtED4zHNwDoDDUMe9YGNhY6zaQQ21XuX8w+kENJ6M\nZHmNtWDQFGNhO7OxwwGwAbzR16POezvHq6r5Ks5AOh0QTaRl0ot5mUjLY6xyALSXGasbXh1NOU2+\n7jke5nOsros2MTgATiDjYYc21vj6iFlzOQBabXr9MO9+GQ917IXxOp/9ONCbCA8gsM4cAG3I+r0D\nMjOrgwOgmfWFD2p6y833I+cAOEHV+ZBu78jMbCJYZ7QrYGZmNhocAM3MrJEcAM3MrJEcAM3MrJHc\nCcbMxhV3wrJe8RmgmZk1ks8AJxAfGZuZdc9ngGZm1kgOgGZm1kgOgGZm1kgOgGZm1kgOgGZm1kgO\ngGZm1kgOgGZm1kgOgGZm1kgOgGZm1kgOgGZm1kgOgGZm1kgOgGZm1kgOgGZm1kijEgAlfUbSPZLu\nlnSBpMmSZki6WdIDkn4oaf2cd4M8PC9/P71UzvE5/X5J7x+NeTEzs/Gp7wFQ0jTgk8CsiHgTMAk4\nFPgq8PWImAksBo7KoxwFLI6IHYCv53xI2jGP90Zgb+C7kib1c17MzGz8Gq0m0HWBDSWtC2wEPAG8\nF7gkf38u8MH8+cA8TP5+T0nK6RdGxIqIeBiYB+zSp/qbmdk41/cAGBG/Bv4ReJQU+JYAtwHPRsSq\nnG0BMC1/ngY8lsddlfNvXk6vGMfMzKyt0WgCfRXp7G0GsDXwCmCfiqzF/5trkO8GS6+a5tGS5kia\ns3DhwqFX2szMJpzRaAJ9H/BwRCyMiJXAj4C3A1NykyjANsDj+fMCYFuA/P2mwKJyesU4a4mIMyNi\nVkTMmjp1aq/nx8zMxqHRCICPArtJ2ihfy9sTuBe4Djg45zkcuDR/viwPk7//WURETj809xKdAcwE\nbunTPJiZ2Ti3bucsvRURN0u6BLgdWAX8AjgT+DFwoaQv57Sz8ihnAedLmkc68zs0l3OPpItIwXMV\ncExErO7rzJiZ2bjV9wAIEBEnAie2JD9ERS/OiHgBOGSQck4FTu15Bc3MbMLzk2DMzKyRHADNzKyR\nHADNzKyRHADNzKyROnaCkTSZ9DzONwKTi/SI+PMa62VmZlarbs4Azwe2BN4PXE+64XxpnZUyMzOr\nWzcBcIeI+BLw24g4F/gA8OZ6q2VmZlavbgLgyvz+rKQ3kR5FNr22GpmZmfVBNzfCn5kfYP1F0uPH\nNga+VGutzMzMatZNALw2IhYDNwDbA+Rnb5qZmY1b3TSB/ntF2iUVaWZmZuPGoGeAkl5PuvVhU0kH\nlb7ahNLtEGZmZuNRuybQ1wH7AVOA/UvpS4G/rLNSZmZmdRs0AEbEpcClknaPiBv7WCczM7PaddMJ\n5heSjsFPgjEzswnET4IxM7NG8pNgzMyskfwkGDMza6ShPAnmSww8Cebvaq2VmZlZzToGwIj4fv54\nPflJMGZmZuNduxvhP9tuxIj4Wu+rY2Zm1h/tzgBfmd9fB7yN1PwJ6ab4G+qslJmZWd3a3Qj/9wCS\nfgrsHBFL8/BJwMV9qZ2ZmVlNuukFuh3wYmn4RdwL1MzMxrlueoGeD9wi6T+AAD4EnFtrrczMzGrW\nTS/QUyX9BHhXTjoyIn5Rb7XMzMzq1c0ZIBFxO3B7zXUxMzPrm26uAZqZmU04DoBmZtZIDoBmZtZI\nHQOgpIMkPSBpiaTnJC2V9Fw/KmdmZlaXbjrBnA7sHxFz666MmZlZv3TTBPpUr4OfpCmSLpF0n6S5\nknaXtJmka/LZ5jX5HyhQ8i1J8yTdKWnnUjmH5/wPSDq8l3U0M7OJrZsAOEfSDyUdlptDD5J00Ain\n+03gqoh4PfAWYC5wHHBtRMwErs3DAPsAM/PraOB7AJI2A04EdgV2AU4sgqaZmVkn3TSBbgIsB/Yq\npQXwo+FMUNImwLuBIwAi4kXgRUkHAnvkbOcCs4EvAAcC50VEADfls8etct5rImJRLvcaYG/gguHU\ny8zMmqWbJ8Ec2eNpbg8sBP5F0luA24BPAa+JiCfyNJ+Q9OqcfxrwWGn8BTltsPSXkXQ06eyR7bbb\nrndzYmZm41a7/wP8m4g4XdIZpDO+tUTEJ0cwzZ2BYyPiZknfZKC5s7IqFWnRJv3liRFnAmcCzJo1\nqzKPmZk1S7szwKLjy5weT3MBsCAibs7Dl5AC4FOStspnf1sBT5fyb1safxvg8Zy+R0v67B7X1czM\nJqh2/wd4eX7v6T8/RMSTkh6T9LqIuB/YE7g3vw4HTsvvl+ZRLgM+IelCUoeXJTlIXg18pdTxZS/g\n+F7W1czMJq6O1wAlTSV1RtkRmFykR8R7RzDdY4EfSFofeAg4ktQj9SJJRwGPAofkvFcC+wLzSJ1x\njszTXyTpFODWnO/kokOMmZlZJ930Av0B8EPgA8DHSGdnC0cy0Yi4A5hV8dWeFXkDOGaQcs4Gzh5J\nXczMrJm6uQ9w84g4C1gZEddHxJ8Du9VcLzMzs1p1cwa4Mr8/IekDpA4o29RXJTMzs/p1EwC/LGlT\n4HPAGaQb4z9Ta63MzMxq1jYASpoEzIyIK4AlwHv6UiszM7Oatb0GGBGrgQP6VBczM7O+6aYJ9H8l\nfZvUE/S3RWJE3F5brczMzGrWTQB8e34/uZQWwEjuAzQzMxtV3QTAoyLioXKCpO1rqo+ZmVlfdHMf\n4CUVaRf3uiJmZmb91O7fIF4PvBHYtOUPcDeh9Eg0MzOz8ahdE+jrgP2AKcD+pfSlwF/WWSkzM7O6\ntfs3iEuBSyXtHhE39rFOZmZmtet4DdDBz8zMJqJuOsGYmZlNOIMGQEmfyu/v6F91zMzM+qPdGeCR\n+f2MflTEzMysn9r1Ap0raT4wVdKdpXSR/qf292qtmZmZWY3a9QI9TNKWwNX4gdhmZjbBtH0UWkQ8\nCbxF0vrA7+bk+yNiZZvRzMzMxryOzwKV9AfAecB8UvPntpIOj4gbaq6bmZlZbbp5GPbXgL0i4n4A\nSb8LXAC8tc6KmZmZ1amb+wDXK4IfQET8ClivviqZmZnVr5szwDmSzgLOz8MfBm6rr0pmZmb16yYA\nfhw4Bvgk6RrgDcB366yUmZlZ3ToGwIhYQboO+LX6q2NmZtYffhaomZk1kgOgmZk1UscAKOll//4u\naYt6qmNmZtYf3ZwB3ippt2JA0h8B/1tflczMzOrXTS/QPwXOljQb2BrYHHhvnZUyMzOrWze9QO+S\ndCrpPsClwLsjYkHtNTMzM6tRN9cAzwI+Dfwe6T8CL5d0zEgnLGmSpF9IuiIPz5B0s6QHJP0wP4Ab\nSRvk4Xn5++mlMo7P6fdLev9I62RmZs3RzTXAu4H3RMTDEXE1sBuwcw+m/Slgbmn4q8DXI2ImsBg4\nKqcfBSyOiB2Ar+d8SNoROBR4I7A38F1Jk3pQLzMza4COATAivg5MlvS6PLwkIo7qMFpbkrYBPgB8\nPw+LdF3xkpzlXOCD+fOBeZj8/Z45/4HAhRGxIiIeBuYBu4ykXmZm1hzdNIHuD9wBXJWHd5J02Qin\n+w3gb4A1eXhz4NmIWJWHFwDT8udpwGMA+fslOf9L6RXjmJmZtdVNE+hJpDOrZwEi4g5gxnAnKGk/\n4OmIKD9QWxVZo8N37cZpnebRkuZImrNw4cIh1dfMzCambgLgqohY0pJWGWi69A7gAEnzgQtJTZ/f\nAKZIKnqlbgM8nj8vALYFyN9vCiwqp1eMs3ZlI86MiFkRMWvq1KkjqLqZmU0UXXWCkfSnwCRJMyWd\nwQhuhI+I4yNim4iYTurE8rOI+DBwHXBwznY4cGn+fFkeJn//s4iInH5o7iU6A5gJ3DLcepmZWbN0\nEwCPJfW0XEH6J/jnSLdF9NoXgM9Kmke6xndWTj8L2DynfxY4DiAi7gEuAu4lXZ88JiJW11AvMzOb\ngLq5EX45cEJ+9VREzAZm588PUdGLMyJeAA4ZZPxTgVN7XS8zM5v4Bg2Aki6nzbW+iDiglhqZmZn1\nQbszwH/M7wcBWwL/mocPA+bXWCczM7PaDRoAI+J6AEmnRMS7S19dLumG2mtmZmZWo246wUyVtH0x\nkHtc+l4CMzMb17r5O6TPALMlPZSHpwN/VVuNzMzM+qCbXqBXSZoJvD4n3RcRK+qtlpmZWb26OQME\neCvpzG9d4C2SiIjzaquVmZlZzToGQEnnA68lPRC7uNE8AAdAMzMbt7o5A5wF7JgfP2ZmZjYhdPuH\nuFvWXREzM7N+6uYMcAvgXkm3kJ4HCvhJMGZmNr51EwBPqrsSZmZm/dbNbRDX96MiZmZm/dTuYdhL\nqX4YtoCIiE1qq5WZmVnN2j0L9JX9rIiZmVk/ddML1MzMbMJxADQzs0ZyADQzs0ZyADQzs0ZyADQz\ns0ZyADQzs0ZyADQzs0ZyADQzs0ZyADQzs0ZyADQzs0ZyADQzs0ZyADQzs0ZyADQzs0ZyADQzs0Zy\nADQzs0ZyADQzs0ZyADQzs0bqewCUtK2k6yTNlXSPpE/l9M0kXSPpgfz+qpwuSd+SNE/SnZJ2LpV1\neM7/gKTD+z0vZmY2fo3GGeAq4HMR8QZgN+AYSTsCxwHXRsRM4No8DLAPMDO/jga+BylgAicCuwK7\nACcWQdPMzKyTvgfAiHgiIm7Pn5cCc4FpwIHAuTnbucAH8+cDgfMiuQmYImkr4P3ANRGxKCIWA9cA\ne/dxVszMbBwb1WuAkqYDvw/cDLwmIp6AFCSBV+ds04DHSqMtyGmDpVdN52hJcyTNWbhwYS9nwczM\nxqlRC4CSNgb+Hfh0RDzXLmtFWrRJf3lixJkRMSsiZk2dOnXolTUzswlnVAKgpPVIwe8HEfGjnPxU\nbtokvz+d0xcA25ZG3wZ4vE26mZlZR6PRC1TAWcDciPha6avLgKIn5+HApaX0j+beoLsBS3IT6dXA\nXpJelTu/7JXTzMzMOlp3FKb5DuAjwF2S7shpfwucBlwk6SjgUeCQ/N2VwL7APGA5cCRARCySdApw\na853ckQs6s8smJnZeNf3ABgRP6f6+h3AnhX5AzhmkLLOBs7uXe3MzKwp/CQYMzNrJAdAMzNrJAdA\nMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNr\nJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdA\nMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNrJAdAMzNr\npHEfACXtLel+SfMkHTfa9TEzs/FhXAdASZOA7wD7ADsCh0nacXRrZWZm48G4DoDALsC8iHgoIl4E\nLgQOHOU6mZnZODDeA+A04LHS8IKcZmZm1pYiYrTrMGySDgHeHxF/kYc/AuwSEce25DsaODoPvg64\nfwST3QJ4pvTeKW044/Q6rYl1aOI8j4U6NHGex0IdhjLOcPxOREwdwfhjU0SM2xewO3B1afh44Pia\npzmn/N4pbTjj9DqtiXVo4jyPhTo0cZ7HQh2GMo5fA6/x3gR6KzBT0gxJ6wOHApeNcp3MzGwcWHe0\nKzASEbFK0ieAq4FJwNkRcc8oV8vMzMaBcR0AASLiSuDKPk7yzJb3TmnDGafXaU2sQ7+n5zqMzvRc\nh6GPY9m47gRjZmY2XOP9GqCZmdmwjPsAKOkQSXMlXZeHL5B0p6TPjLDcKyVNqRqWNF3S3aXvpku6\nW9I7Jb2Q0/aQdEX+PFvSrJbyl0k6VNJzFdOeLelySQeXx8+Pe1slaYqkpcX3LeMeKunB0vAJkha0\nmc89JD0m6WJJ7yulf0zSJyVdkoePkPScpNMlnS3poWKeJC2Q9FFJy1rKPknS5yumeZKkByWtlLRQ\n0j3F9CV9X9Inyk/0kXROaVm8tFwrltms1vR2JO0kad+WtJemNcSy1lonOn03SNpvi7TysszLd26b\naS+TND9vB1eU0vdoGX5pGUm6Q9Ijkj4v6QBJx0naQNJ/5e/+JOebIumvW+r9p6XhP5T08y6Wz02S\n5uXPL60Xkp6R9MP8+RuSnsifb5N0oqQrch1+nOt6hKRvDzKN3+S6Xy7p4Jz3CpUekShpuaTtcx1W\n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\nyNaqAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1d5934038d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "labels = np.loadtxt(open(LABELS, \"rb\"), dtype=str, delimiter=\",\", skiprows=1, usecols = 1)\n",
    "features = np.loadtxt(open(\"data/data.csv\", \"rb\"), dtype=int, delimiter=\",\", skiprows=1, usecols = 0)\n",
    "#plt.figure(figsize=(120,50))\n",
    "#plt.figure(1)\n",
    "#plt.plot(labels,features, 'bo', markersize=20)\n",
    "plt.bar(labels,features, alpha=0.8)\n",
    "#plt.scatter(labels,features, c='r')\n",
    "plt.xlabel(\"Breed\")\n",
    "plt.ylabel(\"Index of train data\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.\n",
      "  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
