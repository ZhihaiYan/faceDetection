/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 * Tue Oct  7 08:12:06 EDT 1997, bthom, added a few comments,
 *       tagged w/bthom
 *
 ******************************************************************
 */

#include <stdio.h>
#include <math.h>
#include "pgmimage.h"
#include "backprop.h"


 //extern char *strcpy();
 //extern void exit();


int main(argc, argv)
int argc;
char *argv[];
{
	char netname[256], trainname[256], test1name[256], test2name[256];
	IMAGELIST *trainlist, *test1list, *test2list;
	int ind, epochs, seed, savedelta, list_errors;

	seed = 102194;   /*** today's date seemed like a good default ***/
	epochs = 100;
	savedelta = 100;
	list_errors = 0;
	netname[0] = trainname[0] = test1name[0] = test2name[0] = '\0';


	argc = 4;

	char options_yzh[20][50] = {
		"-n", "shades.net" ,
		"-e", "50000" ,
		//"-s", "" ,
		//"-S", "" ,
		"-t", "trainset//smile_train.list" ,
		"-1", "trainset//smile_test1.list" ,
		//"-2", "trainset//straightrnd_test2.list" ,
		//"-T", "" ,
	};
	/*** Create imagelists ***/
	trainlist = imgl_alloc();
	test1list = imgl_alloc();
	test2list = imgl_alloc();

	/*** Scan command line ***/
	for (ind = 0; ind < 2 * argc + 1; ind++) {

		/*** Parse switches ***/
		if (options_yzh[ind][0] == '-') {
			switch (options_yzh[ind][1]) {
			case 'n': strcpy(netname, options_yzh[++ind]);
				break;
			case 'e': epochs = atoi(options_yzh[++ind]);
				break;
			case 's': seed = atoi(options_yzh[++ind]);
				break;
			case 'S': savedelta = atoi(options_yzh[++ind]);
				break;
			case 't': strcpy(trainname, options_yzh[++ind]);
				break;
			case '1': strcpy(test1name, options_yzh[++ind]);
				break;
			case '2': strcpy(test2name, options_yzh[++ind]);
				break;
			case 'T': list_errors = 1;
				epochs = 0;
				break;
			default: printf("Unknown switch '%c'\n", options_yzh[ind][1]);
				break;
			}
		}
	}
	strcpy(trainname, "trainset//all_train.list");
	strcpy(test1name, "trainset//all_test1.list");
	strcpy(test2name, "trainset//all_test2.list");
	//strcpy(test2name, "trainset//yzh_test.list");
	/*** If any train, test1, or test2 sets have been specified, then
		 load them in. ***/
	if (trainname[0] != '\0')
		imgl_load_images_from_textfile(trainlist, trainname);
	if (test1name[0] != '\0')
		imgl_load_images_from_textfile(test1list, test1name);
	if (test2name[0] != '\0')
		imgl_load_images_from_textfile(test2list, test2name);

	/*** If we haven't specified a network save file, we should... ***/
	if (netname[0] == '\0') {
		printf("%s: Must specify an output file, i.e., -n <network file>\n",
			argv[0]);
		//exit (-1);
	}

	/*** Don't try to train if there's no training data ***/
	if (trainname[0] == '\0') {
		epochs = 0;
	}

	/*** Initialize the neural net package ***/
	bpnn_initialize(seed);

	/*** Show number of images in train, test1, test2 ***/
	printf("%d images in training set\n", trainlist->n);
	printf("%d images in test1 set\n", test1list->n);
	printf("%d images in test2 set\n", test2list->n);

	/*** If we've got at least one image to train on, go train the net ***/
	backprop_face(trainlist, test1list, test2list, epochs, savedelta, netname,
		list_errors);

	exit(0);
}


backprop_face(trainlist, test1list, test2list, epochs, savedelta, netname,
	list_errors)
	IMAGELIST *trainlist, *test1list, *test2list;
int epochs, savedelta, list_errors;
char *netname;
{
	IMAGE *iimg;
	BPNN *net;
	int train_n, epoch, i, imgsize;
	double out_err, hid_err, sumerr;

	train_n = trainlist->n;

	/*** Read network in if it exists, otherwise make one from scratch ***/
	if ((net = bpnn_read(netname)) == NULL) {
		if (train_n > 0) {
			printf("Creating new network '%s'\n", netname);
			iimg = trainlist->list[0];
			imgsize = ROWS(iimg) * COLS(iimg);
			/* bthom ===========================
		  make a net with:
			imgsize inputs, 4 hiden units, and 1 output unit
				*/
			net = bpnn_create(imgsize, 4, 1);
		}
		else {
			printf("Need some images to train on, use -t\n");
			return;
		}
	}

	if (epochs > 0) {
		printf("Training underway (going to %d epochs)\n", epochs);
		printf("Will save network every %d epochs\n", savedelta);
		fflush(stdout);
	}

	/*** Print out performance before any epochs have been completed. ***/
	printf("0 0.0 ");
	performance_on_imagelist(net, trainlist, 0);
	performance_on_imagelist(net, test1list, 0);
	performance_on_imagelist(net, test2list, 0);
	printf("\n");  fflush(stdout);
	if (list_errors) {
		printf("\nFailed to classify the following images from the training set:\n");
		performance_on_imagelist(net, trainlist, 1);
		printf("\nFailed to classify the following images from the test set 1:\n");
		performance_on_imagelist(net, test1list, 1);
		printf("\nFailed to classify the following images from the test set 2:\n");
		performance_on_imagelist(net, test2list, 1);
	}

	/************** Train it *****************************/
	for (epoch = 1; epoch <= epochs; epoch++)
	{

		

		sumerr = 0.0;
		for (i = 0; i < train_n; i++) {

			/** Set up input units on net with image i **/
			//输入是图像的像素，值为0-1
			load_input_with_image(trainlist->list[i], net);

			/** Set up target vector for image i **/
			//我是0.9，其他人是0.1
			load_target(trainlist->list[i], net);

			/** Run backprop, learning rate 0.3, momentum 0.3 **/
			bpnn_train(net, 0.01, 0.001, &out_err, &hid_err);

			sumerr += (out_err + hid_err);
		}
	

		/*** Evaluate performance on train, test, test2, and print perf ***/
		if (!(epoch % 10)) 
		{
			printf("%d ", epoch);
			printf("%g ", sumerr);
			performance_on_imagelist(net, trainlist, 0);
			performance_on_imagelist(net, test1list, 0);
			performance_on_imagelist(net, test2list, 0);
			printf("\n");  //fflush(stdout);
		}

		/*** Save network every 'savedelta' epochs ***/
		if (!(epoch % savedelta)) {
			//bpnn_save(net, netname);
		}

	}
	//printf("\n"); fflush(stdout);

	/** Save the trained network **/
	if (epochs > 0) {
		bpnn_save(net, netname);
	}
}


/*** Computes the performance of a net on the images in the imagelist. ***/
/*** Prints out the percentage correct on the image set, and the
	 average error between the target and the output units for the set. ***/
performance_on_imagelist(net, il, list_errors)
BPNN *net;
IMAGELIST *il;
int list_errors;
{
	double err, val;
	int i, n, j, correct, itisme,me;

	me = 0;
	err = 0.0;
	correct = 0;
	itisme = 0;
	n = il->n;
	if (n > 0)
	{
		for (i = 0; i < n; i++)
		{

			/*** Load the image into the input layer. **/
			load_input_with_image(il->list[i], net);
			
			/*** Run the net on this input. **/
			bpnn_feedforward(net);
			
			/*** Set up the target vector for this image. **/
			load_target(il->list[i], net);
			if (net->target[1] > 0.5)
			{
				me++;
			}
			/*** See if it got it right. ***/
			//返回值是0或者1，表示是否预测正确
			int aaa = evaluate_performance(net, &val);
			if (aaa != 0)//预测正确
			{
				correct++;
			}
			if (aaa == 2)
			{
				itisme++;
			}
			else if (list_errors)
			{
				printf("%s - outputs ", NAME(il->list[i]));
				for (j = 1; j <= net->output_n; j++)
				{
					printf("%.3f ", net->output_units[j]);
				}
				putchar('\n');
			}
			err += val;
		}

		err = err / (double)n;

		if (!list_errors)
			/* bthom==================================
		   this line prints part of the ouput line
		   discussed in section 3.1.2 of homework
				*/
			printf("%g %g ", ((double)correct / (double)n) *100.0, (double)itisme / (double)me *100.0);// err);
	}
	else
	{
		if (!list_errors)
			printf("0.0 0.0 ");
	}
}

// 输出2 表示检测到我自己
evaluate_performance(net, err)
BPNN *net;
double *err;
{
	double delta;

	delta = net->target[1] - net->output_units[1];

	*err = (0.5 * delta * delta);

	/*** If the target unit is on... ***/
	if (net->target[1] > 0.5) {

		/*** If the output unit is on, then we correctly recognized me! ***/
		if (net->output_units[1] >0.5) {
			return (2);

			/*** otherwise, we didn't think it was me... ***/
		}
		else {
			return (0);
		}

		/*** Else, the target unit is off... ***/
	}
	else {

		/*** If the output unit is on, then we mistakenly thought it was me ***/
		if (net->output_units[1] > 0.5) {
			return (0);

			/*** else, we correctly realized that it wasn't me ***/
		}
		else {
			return (1);
		}
	}

}



printusage(prog)
char *prog;
{
	printf("USAGE: %s\n", prog);
	printf("       -n <network file>\n");
	printf("       [-e <number of epochs>]\n");
	printf("       [-s <random number generator seed>]\n");
	printf("       [-S <number of epochs between saves of network>]\n");
	printf("       [-t <training set list>]\n");
	printf("       [-1 <testing set 1 list>]\n");
	printf("       [-2 <testing set 2 list>]\n");
	printf("       [-T]\n");
}
