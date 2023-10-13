def eval(model, data_loader, criterion, writer, global_step, device):
    model.eval()
    correct = 0
    total = 0
    eval_losses = []

    # Iterate through the evaluation DataLoader
    for k, (eval_lidx, eval_ridx, eval_images, eval_labels) in enumerate(data_loader):
        eval_images = eval_images.squeeze(0).to(device)
        eval_labels = eval_labels.squeeze().to(device).float()

        # Forward pass
        eval_outputs = model(eval_images)
        eval_loss = criterion(eval_outputs.squeeze(), eval_labels)
        eval_losses.append(eval_loss.item())

        # Calculate accuracy
        predicted = (eval_outputs > 0.5).float()
        predicted = predicted.squeeze(1)
        correct += (predicted == eval_labels).sum().item()

        total += len(eval_labels)

        # print(predicted == eval_labels)
        # print(correct, total)

        # Print evaluation results every 5 batches
        if (k + 1) % 5 == 0:
            print(f'Eval Batch [{k + 1}/{len(data_loader)}], Eval Loss: {sum(eval_losses) / len(eval_losses):.4f}')

    # Log and print total evaluation results
    accuracy = 100 * correct / total
    print(f'Evaluation Accuracy: {accuracy:.2f}%, Eval Loss: {sum(eval_losses) / len(eval_losses):.4f}')
    writer.add_scalar('evaluation accuracy', accuracy, global_step)
    writer.add_scalar('evaluation loss', sum(eval_losses) / len(eval_losses), global_step)

    # Switch back to training mode
    model.train()
    return accuracy, sum(eval_losses) / len(eval_losses)
